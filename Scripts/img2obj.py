import gradio as gr

from rembg.processor import check_input_image, preprocess
from triposr.file_io import check_cutout_image, update_triposr_model_filenames
from triposr.processor import generate

from modules import script_callbacks

triposr_models = update_triposr_model_filenames()

def on_ui_tabs():
    with gr.Blocks() as model_block:
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    with gr.Tabs():
                        with gr.Tab("Input Image"):
                            input_image = gr.Image(
                                image_mode="RGBA",
                                sources="upload",
                                type="pil",
                                elem_id="content_image",
                                show_label=False,
                            )
                    with gr.Tabs():
                        with gr.Tab("Processed Image"):
                            processed_image = gr.Image(
                                image_mode="RGBA",
                                sources="upload",
                                type="pil",
                                elem_id="cutout_image",
                                show_label=False,
                            )
                        with gr.Tab("Processed Mask"):
                            processed_image = gr.Image(
                                image_mode="RGBA",
                                sources="upload",
                                type="pil",
                                elem_id="mask_image",
                                show_label=False,
                            )
                        
                        
                with gr.Row():
                    with gr.Column():
                        with gr.Tabs():
                            with gr.Tab("Preprocess Settings"):
                                submit_preprocess = gr.Button(
                                    "Preprocess", 
                                    elem_id="preprocess", 
                                    variant="secondary"
                                )

                                rembg_model_dropdown = gr.Dropdown(
                                    label="Cutout Model",
                                    # choices=get_rembg_model_choices(),
                                    value="dis_general_use",  # Default value
                                )
                                do_remove_background = gr.Checkbox(
                                    label="Remove Background", 
                                    value=True
                                )
                                foreground_ratio = gr.Slider(
                                    label="Subject Zoom",
                                    minimum=0.5,
                                    maximum=1.0,
                                    value=0.85,
                                    step=0.05,
                                )
                                alpha_matting = gr.Checkbox(
                                    label="Enable Alpha Matting", 
                                    value=False
                                )
                                gr.Markdown("*Improves edge and transparency handling*")
                                alpha_matting_foreground_threshold = gr.Slider(
                                    label="Alpha Matting Foreground Threshold",
                                    minimum=0,
                                    maximum=255,
                                    value=240,
                                    step=1,
                                )
                                alpha_matting_background_threshold = gr.Slider(
                                    label="Alpha Matting Background Threshold",
                                    minimum=0,
                                    maximum=255,
                                    value=10,
                                    step=1,
                                )
                                alpha_matting_erode_size = gr.Slider(
                                    label="Alpha Matting Erode Size",
                                    minimum=0,
                                    maximum=50,
                                    value=0,
                                    step=1,
                                )
                    with gr.Column():
                        with gr.Tabs():
                            with gr.Tab("TripoSR"):
                                triposr_render = gr.Button(
                                    "Render", 
                                    elem_id="triposr_render", 
                                    variant="secondary"
                                )
                                triposr_filename = gr.Dropdown(
                                    label="TripoSR Checkpoint",
                                    choices=triposr_models,
                                    value=triposr_models[0] if len(triposr_models) > 0 else None
                                )
                                triposr_resolution = gr.Slider(
                                    label="Mesh Resolution",
                                    minimum=16,
                                    maximum=2048,
                                    value=256,
                                    step=16,
                                )
                                triposr_threshold = gr.Slider(
                                    label="Threshold",
                                    minimum=0,
                                    maximum=100,
                                    value=25,
                                    step=0.1,
                                )
                                triposr_chunking = gr.Slider( #- FIXME - Currently does nothing. Does it actually do anything at all? I don't know. It doesn't appear to affect much in tests.
                                    label="Chunking",
                                    minimum=128,
                                    maximum=16384,
                                    value=8192,
                                    step=128,
                                )
                                triposr_auto_unload = gr.Checkbox(
                                    label="Automatically unload model after generation", 
                                    value=True
                                )
                                triposr_unload = gr.Button(
                                    "Unload TripoSR Model", 
                                    elem_id="triposr_unload", 
                                    variant="secondary"
                                )
                            with gr.Tab("CRM"):
                                crm_render = gr.Button(
                                    "Render", 
                                    elem_id="crm_render", 
                                    variant="secondary"
                                )
                                crm_filename = gr.Dropdown(
                                    label="CRM Checkpoint",
                                    # choices=triposr_model_filenames,
                                    # value=triposr_model_filenames[0] if len(triposr_model_filenames) > 0 else None
                                )
                                crm_steps = gr.Slider(
                                    label="Steps",
                                    minimum=0,
                                    maximum=150,
                                    value=25,
                                    step=1,
                                )
                                crm_cfg = gr.Slider(
                                    label="CFG Scale",
                                    minimum=0,
                                    maximum=100,
                                    value=25,
                                    step=0.1,
                                )
                                crm_seed = gr.Number(value=1234, label="Seed", precision=0)
                                crm_auto_unload = gr.Checkbox(
                                    label="Automatically unload model after generation", 
                                    value=True
                                )
                                crm_unload = gr.Button(
                                    "Unload CRM Model", 
                                    elem_id="crm_unload", 
                                    variant="secondary"
                                )
                        
            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("TripoSR Result"):
                        output_model = gr.Model3D(
                            label="Output Model",
                            interactive=False,
                            elem_id="triposrCanvas",
                            show_label=False,
                        )

                    with gr.Tab("PoSR"):
                        gr.HTML('''
                            <canvas id="babylonCanvas"></canvas>
                        ''')

                        model_block.load(
                            _js = '''
                                function babylonCanvasLoader() {                                
                                    let babylon_script = document.createElement('script');       
                                    babylon_script.src = 'https://cdn.babylonjs.com/babylon.js';
                                    babylon_script.onload = function(){
                                        let babylon_loaders_script = document.createElement('script');       
                                        babylon_loaders_script.src = 'https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js';
                                        babylon_loaders_script.onload = function(){
                                            // Access OBJFileLoader through the BABYLON namespace and enable vertex colors
                                            BABYLON.OBJFileLoader.IMPORT_VERTEX_COLORS = true;
                                           
                                            let babylonCanvasScript = document.createElement('script');
                                            babylonCanvasScript.innerHTML = `
                                                var canvas = document.getElementById('babylonCanvas');
                                                canvas.addEventListener('wheel', function(event) {
                                                    event.preventDefault();
                                                }, { passive: false });
                                           
                                                var engine = new BABYLON.Engine(canvas, true);
                                                var camera; 
                                                var scene
                                           
                                                function createScene(objFile) {
                                                    // Check if a scene already exists and dispose of it if it does
                                                    if (window.scene) {
                                                        window.scene.dispose();
                                                    }

                                                    scene = new BABYLON.Scene(engine);
                                                    scene.clearColor = new BABYLON.Color3.White();
                                           
                                                    camera = new BABYLON.ArcRotateCamera("camera", -Math.PI / 2, Math.PI / 2.5, 10, new BABYLON.Vector3(0, 0, 0), scene, 0.1, 10000);
                                                    camera.attachControl(canvas, true);
                                                    camera.wheelPrecision = 50;
                                           
                                                    var light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 0), scene);
                                                    light.intensity = 1;
                                           
                                                    // Initialize GizmoManager here
                                                    var gizmoManager = new BABYLON.GizmoManager(scene);
                                                    gizmoManager.positionGizmoEnabled = true;
                                                    gizmoManager.rotationGizmoEnabled = true;
                                                    gizmoManager.scaleRatio = 2
                                           
                                                    // Load the OBJ file
                                                    BABYLON.SceneLoader.ImportMesh("", "", "file=" + objFile, scene, function (newMeshes) {
                                                        //camera.target = newMeshes[0];
                                                        camera.target = new BABYLON.Vector3(0, 0, 0); // Keeps the camera focused on the origin
                                                      
                                                        // Define your desired scale factor
                                                        var scaleFactor = 8; // Example: Scale up by a factor of 2
                                                        // Apply a material to all loaded meshes that uses vertex colors
                                                        newMeshes.forEach(mesh => {
                                                            mesh.scaling = new BABYLON.Vector3(scaleFactor, scaleFactor, scaleFactor);
                                                        });
                                                        // Attach the first loaded mesh to the GizmoManager
                                                        if(newMeshes.length > 0) {
                                                            gizmoManager.attachToMesh(newMeshes[0]);
                                                        }
                                                    });
                                           
                                                    return scene;
                                                };
                                           
                                                window.addEventListener('resize', function() {
                                                    engine.resize(); 
                                                });
                                            `
                                            document.head.appendChild(babylonCanvasScript);
                                        };    
                                        document.head.appendChild(babylon_loaders_script);
                                    };    
                                    document.head.appendChild(babylon_script);
                                           
                                    let babylonCanvasStyle = document.createElement('style');
                                    babylonCanvasStyle.innerHTML = `
                                        #babylonCanvas {
                                            width: 100%;
                                            height: 100%;
                                            touch-action: none;
                                        }
                                    `
                                    document.head.appendChild(babylonCanvasStyle);
                                }
                            '''
                        )

                        obj_file = gr.File(
                            label="OBJ File",
                            height=84,
                            file_types=['.obj'],
                        )                            

                        obj_file_path = gr.Textbox(
                            visible=False,
                            label="OBJ File Path",
                            elem_id="obj_file_path"
                        )  # Hidden textbox to pass the OBJ file path

                        scene_background_image = gr.Image(
                            label="Scene Background",
                            image_mode="RGBA",
                            sources="upload",
                            type="pil",
                            elem_id="scene_background_image",
                        )

                        load_obj_btn = gr.Button("Load 3D Scene")
                        load_obj_btn.click(
                            None, [obj_file_path], None, _js='''
                                (objFilePath) => { 
                                    createScene(objFilePath);
                                    engine.runRenderLoop(function() {
                                        scene.render();
                                    });
                                    engine.resize(); 
                                }
                            '''
                        )
                        
                        save_png_width = gr.Slider(
                            label="Image Width",
                            minimum=0,
                            maximum=2048,
                            value=512,
                            step=1,
                        )
                        save_png_height = gr.Slider(
                            label="Image Height",
                            minimum=0,
                            maximum=2048,
                            value=512,
                            step=1,
                        )
                               
                        save_png_btn = gr.Button("Save Current View to PNG")
                        save_png_btn.click(
                            None, [obj_file_path, save_png_width, save_png_height], None, _js='''
                                (objFilePath, save_png_width, save_png_height) => { 
                                    // Export to PNG button functionality
                                    BABYLON.Tools.CreateScreenshotUsingRenderTarget(engine, camera, { width: save_png_width, height: save_png_height }, function(data) {
                                        // Create a link and set the URL as the data returned from CreateScreenshot
                                        var link = document.createElement('a');
                                        link.download = 'scene.png';
                                        link.href = data;
                                        link.click();
                                    });
                                }
                            '''
                        )                

            submit_preprocess.click(
                fn=check_input_image, inputs=[input_image]
            ).success(
                fn=preprocess,
                inputs=[
                    input_image, 
                    rembg_model_dropdown,
                    do_remove_background, 
                    foreground_ratio,
                    alpha_matting,
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold,
                    alpha_matting_erode_size
                ],
                outputs=[processed_image]
            )

            triposr_render.click(
                fn=check_cutout_image, inputs=[processed_image]
            ).success(
                fn=generate,
                inputs=[processed_image, triposr_resolution, triposr_threshold],
                outputs=[output_model, obj_file_path]
            )
            
            # submit.click(
            #     fn=check_input_image, inputs=[input_image]
            # ).success(
            #     fn=preprocess,
            #     inputs=[
            #         input_image, 
            #         rembg_model_dropdown,
            #         do_remove_background, 
            #         foreground_ratio,
            #         alpha_matting,
            #         alpha_matting_foreground_threshold,
            #         alpha_matting_background_threshold,
            #         alpha_matting_erode_size
            #     ],
            #     outputs=[processed_image]
            # ).success(
            #     fn=generate,
            #     inputs=[processed_image, triposr_resolution, triposr_threshold],
            #     outputs=[output_model, obj_file_path]
            # )

    return [(model_block, "img2obj", "img2obj")]

script_callbacks.on_ui_tabs(on_ui_tabs)
