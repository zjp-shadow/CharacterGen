"""OpenGL rasterization on AMD for inference (no autograd/CPU renderer)."""
import threading

import moderngl
import numpy as np
import torch
import torch.nn.functional as F

from runtime import get_device, grid_sample

_local = threading.local()


def context():
    if not hasattr(_local, "gl"):
        _local.gl = moderngl.create_standalone_context(require=430)
        renderer = _local.gl.info["GL_RENDERER"]
        if any(name in renderer.lower() for name in ("llvmpipe", "softpipe", "software", "gdi generic")):
            raise RuntimeError(f"A hardware OpenGL GPU is required; found {renderer}")
        _local.program = _local.gl.program(
            vertex_shader='''#version 430
                in vec4 position;
                void main() { gl_Position = position; }
            ''',
            geometry_shader='''#version 430
                layout(triangles) in;
                layout(triangle_strip, max_vertices=3) out;
                out vec3 bary;
                flat out int triangle_id;
                void main() {
                    for(int i=0;i<3;i++) {
                        gl_Position=gl_in[i].gl_Position;
                        bary=vec3(0); bary[i]=1;
                        triangle_id=gl_PrimitiveIDIn+1;
                        EmitVertex();
                    }
                    EndPrimitive();
                }
            ''',
            fragment_shader='''#version 430
                in vec3 bary;
                flat in int triangle_id;
                out vec4 color;
                void main() { color=vec4(bary.xy, gl_FragCoord.z*2-1, float(triangle_id)); }
            ''')
    return _local.gl


class RasterizeCudaContext:
    # Retains the original call signature for the optional back projection pass.
    def __init__(self, device=None):
        self.device = device if device is not None else get_device()


RasterizeGLContext = RasterizeCudaContext


def rasterize(ctx, pos, tri, resolution, **kwargs):
    if torch.is_grad_enabled() and pos.requires_grad:
        raise RuntimeError("The AMD OpenGL renderer supports inference only.")
    gl = context()
    height, width = resolution
    if len(tri) == 0:
        return torch.zeros((pos.shape[0], height, width, 4), device=pos.device), None
    resources = []
    try:
        color = gl.texture((width, height), 4, dtype="f4"); resources.append(color)
        depth = gl.depth_renderbuffer((width, height)); resources.append(depth)
        fbo = gl.framebuffer([color], depth); resources.append(fbo)
        ibo = gl.buffer(tri.detach().cpu().numpy().astype("i4").tobytes()); resources.append(ibo)
        result = []
        fbo.use()
        gl.viewport = (0, 0, width, height)
        gl.enable_only(moderngl.DEPTH_TEST)
        gl.depth_func = "<="
        for vertices in pos.detach().float().cpu().numpy():
            vbo = gl.buffer(vertices.tobytes())
            vao = gl.vertex_array(_local.program, [(vbo, "4f", "position")], ibo)
            try:
                fbo.clear(0, 0, 0, 0, depth=1)
                vao.render(moderngl.TRIANGLES)
                result.append(np.frombuffer(color.read(), np.float32).reshape(height, width, 4).copy())
            finally:
                vao.release(); vbo.release()
        rast = torch.from_numpy(np.stack(result)).to(pos.device)
        return rast, None
    finally:
        for resource in reversed(resources):
            resource.release()


def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    if diff_attrs is not None:
        raise NotImplementedError("Attribute derivatives are not used by AMD inference.")
    if attr.ndim == 2:
        attr = attr.unsqueeze(0)
    if len(tri) == 0:
        return torch.zeros((*rast.shape[:-1], attr.shape[-1]), dtype=attr.dtype, device=attr.device), None
    ids = (rast[..., 3].long() - 1).clamp(min=0)
    weights = torch.stack((rast[..., 0], rast[..., 1], 1-rast[..., 0]-rast[..., 1]), -1)
    indices = tri.long()[ids]
    batches = []
    for batch in range(rast.shape[0]):
        values = attr[batch if attr.shape[0] > 1 else 0][indices[batch]]
        batches.append((values * weights[batch, ..., None]).sum(-2))
    return torch.stack(batches) * (rast[..., 3:] > 0).to(attr.dtype), None


def texture(tex, uv, filter_mode="linear", **kwargs):
    if filter_mode != "linear":
        raise NotImplementedError("AMD back projection uses linear texture filtering.")
    # nvdiffrast's default texture boundary is periodic.
    image = tex.permute(0, 3, 1, 2)
    image = F.pad(image, (1, 1, 1, 1), mode="circular")
    h, w = tex.shape[1:3]
    coords = uv - uv.floor()
    scale = torch.tensor([w/(w+2), h/(h+2)], device=uv.device)
    grid = (coords*2-1)*scale
    return grid_sample(image, grid).permute(0, 2, 3, 1)


def antialias(color, rast, pos, tri):
    # Inference edge smoothing; rasterization and filtering both run on the GPU.
    mask = (rast[..., 3:] > 0).float().permute(0, 3, 1, 2)
    coverage = F.avg_pool2d(mask, 3, stride=1, padding=1)
    image = color.permute(0, 3, 1, 2)
    smooth = F.avg_pool2d(image, 3, stride=1, padding=1)
    edge = (coverage > 0) & (coverage < 1)
    return torch.where(edge, smooth, image).permute(0, 2, 3, 1)


def marching_tetrahedra(positions, level, indices, triangle_table):
    """Extract and weld mesh edges with compute shaders on the OpenGL GPU."""
    gl = context()
    resources = []
    def buffer(data=None, reserve=0):
        obj = gl.buffer(data=data, reserve=reserve)
        resources.append(obj)
        return obj
    try:
        table = triangle_table.detach().cpu().numpy().astype("i4").ravel()
        shader = gl.compute_shader('''#version 430
            layout(local_size_x=256) in;
            layout(std430,binding=0) readonly buffer P { float p[]; };
            layout(std430,binding=1) readonly buffer S { float s[]; };
            layout(std430,binding=2) readonly buffer T { uvec4 t[]; };
            layout(std430,binding=3) buffer C { uint count; };
            layout(std430,binding=4) writeonly buffer V { float v[]; };
            layout(std430,binding=5) writeonly buffer E { uvec2 edges[]; };
            uniform uint num_tets;
            uniform bool count_only;
            const ivec2 pairs[6]=ivec2[6](ivec2(0,1),ivec2(0,2),ivec2(0,3),ivec2(1,2),ivec2(1,3),ivec2(2,3));
            const int table[96]=int[96](''' + ','.join(map(str, table)) + ''');
            void main() {
                uint id=gl_GlobalInvocationID.x;
                if(id>=num_tets) return;
                uvec4 tet=t[id];
                int code=0;
                for(int i=0;i<4;i++) if(s[tet[i]]>0) code|=(1<<i);
                if(code==0 || code==15) return;
                uint n=table[code*6+3]<0 ? 3u:6u;
                uint start=atomicAdd(count,n);
                if(count_only) return;
                for(uint i=0;i<n;i++) {
                    ivec2 pair=pairs[table[code*6+int(i)]];
                    uint a=min(tet[pair.x],tet[pair.y]);
                    uint b=max(tet[pair.x],tet[pair.y]);
                    float ratio=s[a]/(s[a]-s[b]);
                    for(uint d=0;d<3;d++) v[(start+i)*3+d]=mix(p[a*3+d],p[b*3+d],ratio);
                    edges[start+i]=uvec2(a,b);
                }
            }
        '''); resources.append(shader)
        buffer(positions.detach().float().cpu().numpy().tobytes()).bind_to_storage_buffer(0)
        buffer(level.detach().float().cpu().numpy().tobytes()).bind_to_storage_buffer(1)
        buffer(indices.detach().cpu().numpy().astype("u4").tobytes()).bind_to_storage_buffer(2)
        count = buffer(np.zeros(1, "u4").tobytes()); count.bind_to_storage_buffer(3)
        dummy = buffer(reserve=16); dummy.bind_to_storage_buffer(4); dummy.bind_to_storage_buffer(5)
        shader["num_tets"] = len(indices)
        shader["count_only"] = True
        shader.run(group_x=(len(indices)+255)//256)
        gl.memory_barrier()
        size = int(np.frombuffer(count.read(), "u4")[0])
        if size == 0:
            raise RuntimeError("The predicted surface is empty. Try another input image.")
        vertices = buffer(reserve=size*12); vertices.bind_to_storage_buffer(4)
        edges = buffer(reserve=size*8); edges.bind_to_storage_buffer(5)
        count.write(np.zeros(1, "u4").tobytes())
        shader["count_only"] = False
        shader.run(group_x=(len(indices)+255)//256)
        gl.memory_barrier()
        capacity = 1 << (size*2-1).bit_length()
        slots = buffer(np.full(capacity, 0xffffffff, "u4").tobytes()); slots.bind_to_storage_buffer(6)
        faces = buffer(reserve=size*4); faces.bind_to_storage_buffer(7)
        weld = gl.compute_shader('''#version 430
            layout(local_size_x=256) in;
            layout(std430,binding=5) readonly buffer E { uvec2 edges[]; };
            layout(std430,binding=6) buffer H { uint slots[]; };
            layout(std430,binding=7) writeonly buffer F { uint faces[]; };
            uniform uint size;
            uniform uint mask;
            void main() {
                uint i=gl_GlobalInvocationID.x;
                if(i>=size) return;
                uvec2 edge=edges[i];
                uint slot=(edge.x*73856093u ^ edge.y*19349663u)&mask;
                for(uint attempt=0;attempt<=mask;attempt++) {
                    uint found=atomicCompSwap(slots[slot],0xffffffffu,i);
                    if(found==0xffffffffu) { faces[i]=i; return; }
                    if(all(equal(edges[found],edge))) { faces[i]=found; return; }
                    slot=(slot+1u)&mask;
                }
            }
        '''); resources.append(weld)
        weld["size"] = size; weld["mask"] = capacity-1
        weld.run(group_x=(size+255)//256)
        gl.memory_barrier()
        v = torch.from_numpy(np.frombuffer(vertices.read(), "f4").reshape(-1,3).copy()).to(positions.device)
        f = torch.from_numpy(np.frombuffer(faces.read(), "u4").reshape(-1,3).astype("i8")).to(positions.device)
        return v, f
    finally:
        for resource in reversed(resources):
            resource.release()
