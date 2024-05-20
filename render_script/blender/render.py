import bpy, random
import os
import sys
import pdb
import math
from mathutils import Vector

def gc():
    for i in range(10): bpy.ops.outliner.orphans_purge()

def clear():
    [bpy.data.objects.remove(bpy.data.objects[x]) for x in list(bpy.data.objects.keys())]
    gc()

def importVrm(importVrmPath):
    old_objs = set(bpy.context.scene.objects)
    result = bpy.ops.import_scene.vrm(filepath=importVrmPath)
    return [x for x in set(bpy.context.scene.objects)-old_objs if x.type=="ARMATURE"][0]

def importFbx(importFbxPath):
    old_objs = set(bpy.context.scene.objects)
    result = bpy.ops.import_scene.fbx(filepath=importFbxPath)
    return list(set(bpy.context.scene.objects)-old_objs)[0]

def get_keyframes(obj_list):
    keyframes = []
    for obj in obj_list:
        anim = obj.animation_data
        if anim is not None and anim.action is not None:
            for fcu in anim.action.fcurves:
                for keyframe in fcu.keyframe_points:
                    x, y = keyframe.co
                    if x not in keyframes:
                        keyframes.append(int(x))
    return keyframes

def retarget(source_armature,target_armature):
    bpy.context.view_layer.objects.active = source_armature
    bpy.context.scene.source_rig=source_armature.name
    bpy.context.scene.target_rig=target_armature.name
    bpy.ops.arp.build_bones_list()
    bpy.ops.arp.import_config(filepath=os.path.abspath("remap_mixamo.bmap"))
    bpy.ops.arp.auto_scale()
    keyframes=get_keyframes([source_armature])
    
    bpy.ops.arp.retarget(frame_end=int(max(keyframes)))

def look_at(obj_camera, point):
    direction = point - obj_camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj_camera.rotation_euler = rot_quat.to_euler()

def render_4_views(folder, origin = (0, 0, 0)):
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_x = 768
    bpy.context.scene.render.resolution_y = 768

    camera_positions = {
        'front': (0, -2.5, 0.5),
        'back': (0, 2.5, 0.5),
        'left': (-2.5, 0, 0.5),
        'right': (2.5, 0, 0.5),
    }

    camera_data = bpy.data.cameras.new(name='MyCamera')
    camera_data.angle = math.radians(40)
    camera_object = bpy.data.objects.new('MyCamera', camera_data)

    bpy.context.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object

    camera = bpy.data.objects['MyCamera']
    for angle, position in camera_positions.items():
        camera.location = Vector(position) + Vector(origin)
        look_at(camera, Vector(origin))
        
        bpy.context.scene.render.filepath = f'{folder}/{angle}.png'
        
        bpy.ops.render.render(write_still=True)

def changeApose(armature):
    bones = armature.pose.bones
    if "J_Bip_L_UpperArm" in bones:
        L_arm_name = "J_Bip_L_UpperArm" 
        R_arm_name = "J_Bip_R_UpperArm"
        L_leg_name = "J_Bip_L_UpperLeg"
        R_leg_name = "J_Bip_R_UpperLeg"
    elif "腕上_L.002" in bones:
        L_arm_name = "腕上_L.002"
        R_arm_name = "腕上_R.002"
        L_leg_name = "太もも_L.001"
        R_leg_name = "太もも_R.001"
    elif "Left arm" in bones:
        L_arm_name = "Left arm"
        R_arm_name = "Right arm"
        L_leg_name = "Left leg"
        R_leg_name = "Right leg"
    elif "upper_arm.L" in bones:
        L_arm_name = "upper_arm.L"
        R_arm_name = "upper_arm.R"
        L_leg_name = "upper_leg.L"
        R_leg_name = "upper_leg.R"
    elif "LeftArm" in bones:
        L_arm_name = "LeftArm"
        R_arm_name = "RightArm"
        L_leg_name = "LeftUpLeg"
        R_leg_name = "RightUpLeg"
    elif "Arm_L" in bones:
        L_arm_name = "Arm_L"
        R_arm_name = "Arm_R"
        L_leg_name = "UpLeg_L"
        R_leg_name = "UpLeg_R"
    elif "mixamorig:LeftArm" in bones:
        L_arm_name = "mixamorig:LeftArm"
        R_arm_name = "mixamorig:RightArm"
        L_leg_name = "mixamorig:LeftUpLeg"
        R_leg_name = "mixamorig:RightUpLeg"
    elif "UpperArm_L" in bones:
        L_arm_name = "UpperArm_L"
        R_arm_name = "UpperArm_R"
        L_leg_name = "UpperLeg_L"
        R_leg_name = "UpperLeg_R"
    else:
        import pdb; pdb.set_trace()

    if L_arm_name in bones:
        bones[L_arm_name].rotation_mode = "XYZ"
        bones[L_arm_name].rotation_euler = (-math.pi / 4, 0.0, 0.0)
        bones[L_arm_name].keyframe_insert(data_path="rotation_euler",frame=0)

    if R_arm_name in bones:
        bones[R_arm_name].rotation_mode = "XYZ"
        bones[R_arm_name].rotation_euler = (-math.pi / 4, 0.0, 0.0)
        bones[R_arm_name].keyframe_insert(data_path="rotation_euler",frame=0)

    if L_leg_name in bones:
        bones[L_leg_name].rotation_mode = "XYZ"
        bones[L_leg_name].rotation_euler = (-math.pi / 30, 0.0, 0.0)
        bones[L_leg_name].keyframe_insert(data_path="rotation_euler",frame=0)

    if R_leg_name in bones:
        bones[R_leg_name].rotation_mode = "XYZ"
        bones[R_leg_name].rotation_euler = (-math.pi / 30, 0.0, 0.0)
        bones[R_leg_name].keyframe_insert(data_path="rotation_euler",frame=0)


def move_origin_to_center(obj):
    local_bbox_center = 0.125 * sum((Vector(b) for b in obj.bound_box), Vector())
    scale_factor = max(obj.dimensions)
    return local_bbox_center
    #print(local_bbox_center)
   #local_bbox_center = 0.125 * sum((Vector(b) for b in obj.bound_box), Vector())
    #global_bbox_center = obj.matrix_world @ local_bbox_center
  
    # for cur_obj in bpy.context.scene.objects:
    #     if cur_obj.type != "MESH":
    #         continue
    #     print(cur_obj.name, cur_obj.type)
    #     import pdb; pdb.set_trace()
    #     global_bbox_center = local_bbox_center @ cur_obj.matrix_world 
    #     cur_obj.location -= global_bbox_center
        #cur_obj.scale /= scale_factor
        #obj.scale /= max(obj.dimensions)

   #  # bpy.ops.object.select_all(action='DESELECT') 
   #  # cur_obj.select_set(True) 
   #  # bpy.context.view_layer.objects.active = cur_obj
   #  # bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
   
def export(armature,exportFileNamePattern,apose=False,origin=None):
    bpy.ops.object.select_all(action='DESELECT')
    [x.select_set(True) for x in armature.children if(x.type=="MESH")]
    if apose:
        changeApose(armature)
        os.makedirs(folder + "/apose",exist_ok=True)
        bpy.ops.wm.obj_export(filepath=folder + "/apose.obj",export_animation=True,start_frame=0,end_frame=0,
                            export_selected_objects=True,export_materials=False,export_colors=False,export_uv=False,export_normals=False)
        render_4_views(folder + "/apose", origin)
    else:
        keyframes = get_keyframes([armature])
        #rand_frame = int(random.choice(keyframes))
        os.makedirs(folder + "/pose",exist_ok=True)
        bpy.ops.wm.obj_export(filepath=folder + "/pose.obj",export_animation=True,start_frame=0,end_frame=0,
                            export_selected_objects=True,export_materials=False,export_colors=False,export_uv=False,export_normals=False)
        render_4_views(folder + "/pose", origin)

def exportAnimatedMesh(importVrmPath,importFbxPath,folder,apose):
    clear()
    human=importVrm(importVrmPath)
    # resize human
    if apose:
        origin = move_origin_to_center(human)
        export(human, folder, True, origin)
    else:
        anim = importFbx(importFbxPath)
        retarget(anim, human)
        origin = move_origin_to_center(human)
        export(human, folder, False, origin)
   #bpy.data.objects.remove(anim)
   #gc()
   #bpy.data.objects.remove(human)
   #gc()

if(__name__=="__main__"):
    argv = sys.argv
    if("--" in argv):
        argv = argv[argv.index("--") + 1:]
        importVrmPath, importFbxPath, folder, apose=argv
    else:
        raise Exception("no args")
    print("importVrmPath:", importVrmPath)
    exportAnimatedMesh(importVrmPath, importFbxPath, folder, int(apose))