import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin } from './three-vrm.module.js';
import { loadMixamoAnimation } from './loadMixamoAnimation.js';

// renderer
let renderer = null;

import JSZip from 'jszip';
import { OneMinusDstAlphaFactor } from 'three';
import { forEach } from 'jszip';

function initializeRenderer() {
    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true});
    renderer.setClearColor(0x000000, 0.0);
    // renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setSize(768, 768);
    renderer.setPixelRatio(1);
}

// camera
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 20.0);
const exporter = new GLTFExporter();
//camera.position.set(4.0, 0.0, 0.0);

// camera controls
let controls = null;
let azimuth = Math.PI, elevation = 0, distance = 1.5;

function updateControls(pos = {x: 0, y: 0, z: 0}) {
    controls.screenSpacePanning = true;
    controls.target.set(pos.x, pos.y, pos.z);
    controls.enableRotate = true;
    camera.position.set(distance * Math.cos(elevation) * Math.cos(azimuth), 
                        distance * Math.sin(elevation),
                        distance * Math.cos(elevation) * Math.sin(azimuth));
    controls.update();
}

function initializeControls() {
    controls = new OrbitControls(camera, renderer.domElement);
    updateControls();
}

// scene
let scene = null;
const ambientLight = new THREE.AmbientLight(0x404040, 1.0); // soft white light
const light = new THREE.DirectionalLight(0xffffff);

function initializeScene() {
    scene = new THREE.Scene();
    scene.add(ambientLight);
    light.position.set(1.0, 1.0, 1.0).normalize();
    scene.add(light);
}

// gltf and vrm
let currentVrm = undefined;
const loader = new GLTFLoader();

loader.crossOrigin = 'anonymous';

loader.register((parser) => {
    return new VRMLoaderPlugin(parser);
});

// let vrmPaths = [];
// fetch("./vroid.json").then(
//     (response) => {
//         if (!response.ok) {
//             throw new Error("Fetch request failed");
//         }
//         return response.json();
//     }).then((data) => {
//         console.log(data);
//         vrmPaths = data;
//         processNextVrm();
//     });

let vrmPaths = [];

const queryParams = new URLSearchParams(window.location.search);
const jsonFileName = queryParams.get('file') || 'default.json';

fetch(`./${jsonFileName}`).then(
    (response) => {
        if (!response.ok) {
            throw new Error("Fetch request failed");
        }
        return response.json();
    }).then((data) => {
        console.log(data);
        vrmPaths = data;
        console.log("initializeRenderer");
        processNextVrm();
    });

let base_euler = null, euler_array = null, node_arr = null;
let is_apose = false;
let pose_euler = null, bone_arr = null;

function changePose() {
    bone_arr = ["leftUpperArm", "rightUpperArm", 
                    "leftLowerArm", "rightLowerArm", 
                    "leftHand", "rightHand",
                    "leftShoulder", "rightShoulder",
                    "leftUpperLeg", "rightUpperLeg",
                    "leftLowerLeg", "rightLowerLeg",
                    "leftFoot", "rightFoot", "head",
                    "leftIndexProximal", "rightIndexProximal", "leftIndexDistal", "rightIndexDistal",
                    "leftIndexIntermediate", "rightIndexIntermediate", "leftToes", "rightToes",
                    "upperChest", "neck",
                    "hips", "spine"]; 
    if (is_apose) {
        for (var i = 0; i < bone_arr.length; ++i) {
            if (i < 6)
                currentVrm.humanoid.getNormalizedBoneNode(bone_arr[i])?.rotation.copy(euler_array[i]);
            else
                currentVrm.humanoid.getNormalizedBoneNode(bone_arr[i])?.rotation.copy(new THREE.Euler(0, 0, 0, 'XYZ'));
        }
    } else {
        for(var i = 0; i < pose_euler.length; ++i) {
            if (bone_arr.includes(pose_euler[i].name))
                currentVrm.humanoid.getNormalizedBoneNode(pose_euler[i].name)?.rotation.copy(pose_euler[i].euler);
        }
    }
    currentVrm.update(0);

    node_arr = {};

    for (var i = 0; i < bone_arr.length; ++i) {
        var cur_node = currentVrm.humanoid.getNormalizedBoneNode(bone_arr[i]);
        // console.log(bone_arr[i]);
        if (cur_node != null) {
            node_arr[bone_arr[i]] = {
                world_position: cur_node.getWorldPosition(new THREE.Vector3()),
                position: cur_node.position,
                rotation: cur_node.rotation,
                quaternion: cur_node.quaternion
            }
            // console.log(cur_node);
        }
        // console.log(node_arr[bone_arr[i]]);
    }
    // exit();
}

function aPose() {
    is_apose = true;
    let temp_arr = Array(6).fill(null).map(() => new THREE.Euler());
    temp_arr[0] = new THREE.Euler(0, 0, Math.PI / 4, 'XYZ');
    temp_arr[1] = new THREE.Euler(0, 0, -Math.PI / 4, 'XYZ');
    temp_arr[2] = new THREE.Euler(0, 0, 0, 'XYZ');
    temp_arr[3] = new THREE.Euler(0, 0, 0, 'XYZ');
    temp_arr[4] = new THREE.Euler(0, 0, -Math.PI / 30, 'XYZ');
    temp_arr[5] = new THREE.Euler(0, 0, Math.PI / 30, 'XYZ');
    return temp_arr;
}

function randPose() {
    is_apose = false;
    let temp_arr = Array(6).fill(null).map(() => new THREE.Euler());
    for (var i = 0; i < 6; ++i) {
        // randomize
        if (i < 4)
            temp_arr[i] = new THREE.Euler(
                (((Math.random() > 0.8) ^ (i & 1) ? -1 : 1)) * Math.random() * Math.PI / 180 * 30,
                (((Math.random() > 0.5) ^ (i & 1) ? -1 : 1)) * Math.random() * Math.PI / 180 * 30,
                (((Math.random() > 0.5) ^ (i & 1) ? -1 : 1)) * Math.random() * Math.PI / 180 * 50,
                'XYZ'
            )
        else
            temp_arr[i] = new THREE.Euler(
                ((Math.random() > 0.8) ^ (i & 1) ? -1 : 1) * Math.random() * Math.PI / 180 * 10,
                ((Math.random() > 0.5) ^ (i & 1) ? -1 : 1) * Math.random() * Math.PI / 180 * 10,
                ((Math.random() > 0.5) ^ (i & 1) ? -1 : 1) * Math.random() * Math.PI / 180 * 20,
                'XYZ'
            )
    }
    return temp_arr;
}
function normalizeVrm() {
    const box = new THREE.Box3().setFromObject(currentVrm.scene);
    const size = box.getSize(new THREE.Vector3());

    const maxDimension = Math.max(size.x, size.y, size.z);
    const scale = 1 / maxDimension;

    currentVrm.scene.scale.set(scale, scale, scale);

    const center = box.getCenter(new THREE.Vector3());
    currentVrm.scene.position.sub(center.multiplyScalar(scale));
    currentVrm.update(0);
}

function loadVRM(path) {
    loader.load(
        path,
        (gltf) => {
            const vrm = gltf.userData.vrm;
            scene.add(vrm.scene);

            currentVrm = vrm;
            // print vrm
            // console.log(vrm);
            normalizeVrm();
            //var fbx_id = 11;
            var fbx_id = Math.floor(Math.random() * 24);
            loadFBX("animation/test" + fbx_id + ".fbx");
            base_euler = randPose();

            loader.manager.onLoad = () => {
                animate();
            };
        },
        (progress) => { },
        //console.log('Loading model...', 100.0 * (progress.loaded / progress.total), '%'),
        (error) => console.error(error),
    );
}

let currentIndex = 0;
let Vrmname = "";

function loadFBX( animationUrl ) {
    loadMixamoAnimation( animationUrl, currentVrm ).then( ( result ) => {
        pose_euler = result;
    })
}

function processNextVrm() {
    try {
        initializeRenderer();
        console.log("initializeControls");
        initializeControls();
        console.log("initializeScene");
        initializeScene();
        if (currentIndex < vrmPaths.length) {
            Vrmname = vrmPaths[currentIndex].split("/")[2];
            loadVRM(vrmPaths[currentIndex]);
            currentIndex++;
        } else {
            console.log('All VRMs processed.');
            return;
        }
    }
    catch (e) {
        console.log(e);
        processNextVrm();
    }
}

let cache_data = new JSZip();

function releaseCache() {
    console.log("release cache");
    var formData = new FormData();
    cache_data.forEach(function (path, file) {
        if (!file.dir) {
            cache_data.file(path).async('blob').then(function (blob) {
                formData.append('files', blob, path);
            });
        }
    });

    cache_data.generateAsync({ type: "blob" })
        .then(function (content) {
            return fetch('http://localhost:17070/upload/', {
                method: 'POST',
                body: formData
            });
        });
    console.log("cache released!");
    cache_data = new JSZip();
}

function uploadCache(data, filename) {
    cache_data.file(filename, data);
}

function saveScreenshot(id, type) {
    var screenshotDataUrl = renderer.domElement.toDataURL("image/png");


    // Convert DataURL to Blob
    fetch(screenshotDataUrl)
        .then(res => res.blob())
        .then(blob => {
            let updateName = Vrmname + "_" + id.toString().padStart(3, '0');
            uploadCache(blob, updateName + "_" + type + ".png");

            var json = JSON.stringify({
                name: updateName,
                elevation: elevation,
                azimuth: azimuth,
                distance: distance,
                extrinsicMatrix: camera.matrixWorld,
                intrinsicMatrix: camera.projectionMatrix,
                node_array: node_arr
            });
            var json_blob = new Blob([json], { type: 'application/json' });
            uploadCache(json_blob, updateName + ".json");
        })
        .catch((error) => {
            console.error('Error:', error);
        });
}

var releaseRender = function (renderer, scene) {
    let clearScene = function (scene) {
        let arr = scene.children.filter(x => x);
        arr.forEach(item => {
            if (item.children.length) { clearScene(item); }
            else { if (item.type === 'Mesh') { item.geometry.dispose(); item.material.dispose(); !!item.clear && item.clear(); } }
        });
        !!scene.clear && scene.clear(renderer); arr = null;
    }
    try { clearScene(scene); } catch (e) { }
    try {
        renderer.renderLists.dispose();
        renderer.dispose(); renderer.forceContextLoss();
        renderer.domElement = null; renderer.content = null; renderer = null;
    } catch (e) { }
    if (!!window.requestAnimationId) { cancelAnimationFrame(window.requestAnimationId); } THREE.Cache.clear();
}

function removeCurrentVRM() {
    releaseRender(renderer, scene);
}

let frame = 0, param_aa, param_blinkl, param_blinkr;
let start_azim;
function animate() {
    // requestAnimationFrame(animate);
    if (frame == 0) {
        param_aa = Math.random();
        param_blinkl = Math.random();
        param_blinkr = Math.random(); 
    }
    if (frame % 2 == 0) {
        if (frame < 8) {
            elevation = 0;
            distance = 1.5;
            azimuth = Math.PI / 2 * (frame / 2);
        } else if (frame < 32) {
            if (frame % 8 == 0) {
                elevation = (Math.random() - 0.5) * Math.PI / 6;
                start_azim = Math.PI / 2 * (Math.random());
            }
            distance = 1.5;
            azimuth = Math.PI / 2 * ((frame - 8) / 2) + start_azim;
        } else {
            elevation = 0 + (Math.random() - 0.5) * Math.PI / 4;
            distance = 1.5 + (Math.random() - 0.5);
            azimuth = Math.random() * Math.PI * 2;
        }
        updateControls();
        euler_array = aPose();

        currentVrm.expressionManager.setValue('aa', 0);
        currentVrm.expressionManager.setValue('blinkLeft', 0);
        currentVrm.expressionManager.setValue('blinkRight', 0);
    } else {
        if (frame >= 32) {
            elevation = 0 + (Math.random() - 0.5) * Math.PI / 4;
            distance = 1.5 + (Math.random() - 0.5);
            azimuth = Math.random() * Math.PI * 2;
            var jitter = 0.2;
            updateControls({ x: (Math.random() - 0.5) * jitter, y: (Math.random() - 0.5) * jitter, z: (Math.random() - 0.5) * jitter });
        }

        currentVrm.expressionManager.setValue('aa', param_aa);
        currentVrm.expressionManager.setValue('blinkLeft', param_blinkl);
        currentVrm.expressionManager.setValue('blinkRight', param_blinkr);
        is_apose = false;
    }
        changePose();
    //currentMixer.update(0);
    //normalizeVrm();

	function setMToonDebugMode(material, mode) {
		if ( material.isMToonMaterial ) {
			material.debugMode = mode;
		}
	}

    //const debugMode = ['none', 'normal', 'litShadeRate', 'uv'][debugModeIndex];

    if (frame < 60) {
        currentVrm.scene.traverse( ( object ) => {
            if ( object.material ) {
                if ( Array.isArray( object.material ) ) {
                    object.material.forEach( ( material ) => setMToonDebugMode( material, 'normal') );
                } else {
                    setMToonDebugMode( object.material, 'normal');
                }
            }
        } );
        renderer.render(scene, camera);
        saveScreenshot(frame, "normal");

        currentVrm.scene.traverse( ( object ) => {
            if ( object.material ) {
                if ( Array.isArray( object.material ) ) {
                    object.material.forEach( ( material ) => setMToonDebugMode( material, 'none') );
                } else {
                    setMToonDebugMode( object.material, 'none');
                }
            }
        } );
        renderer.render(scene, camera);
        saveScreenshot(frame, "rgb");

        
        currentVrm.scene.traverse( ( object ) => {
            if ( object.material ) {
                if ( Array.isArray( object.material ) ) {
                    object.material.forEach( ( material ) => setMToonDebugMode( material, 'depth') );
                } else {
                    setMToonDebugMode( object.material, 'depth');
                }
            }
        } );
        renderer.render(scene, camera);
        renderer.render(scene, camera);
        saveScreenshot(frame, "depth");

        frame++;
        (async function () {
            await new Promise(resolve => setTimeout(() => {
                requestAnimationFrame(animate);
                resolve();
            }, 100));
        })();
    } else {
        frame = 0;
        removeCurrentVRM();
        releaseCache();
        (async function () {
            await new Promise(resolve => setTimeout(() => {
                processNextVrm();
                resolve();
            }, 2000));
        })();
    }
}