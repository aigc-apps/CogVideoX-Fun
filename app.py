import time 
import torch

from cogvideox.api.api import infer_forward_api, update_diffusion_transformer_api, update_edition_api
from cogvideox.ui.ui import ui_modelscope, ui_eas, ui

if __name__ == "__main__":
    # Choose the ui mode  
    ui_mode = "normal"
    
    # Low gpu memory mode, this is used when the GPU memory is under 16GB
    low_gpu_memory_mode = False
    # Use torch.float16 if GPU does not support torch.bfloat16
    # ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
    weight_dtype = torch.bfloat16

    # Server ip
    server_name = "0.0.0.0"
    server_port = 7860

    # Params below is used when ui_mode = "modelscope"
    model_name = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP"
    # "Inpaint" or "Control"
    model_type = "Inpaint"
    # Save dir of this model
    savedir_sample = "samples"

    if ui_mode == "modelscope":
        demo, controller = ui_modelscope(model_name, model_type, savedir_sample, low_gpu_memory_mode, weight_dtype)
    elif ui_mode == "eas":
        demo, controller = ui_eas(model_name, savedir_sample)
    else:
        demo, controller = ui(low_gpu_memory_mode, weight_dtype)

    # launch gradio
    app, _, _ = demo.queue(status_update_rate=1).launch(
        server_name=server_name,
        server_port=server_port,
        prevent_thread_lock=True
    )
    
    # launch api
    infer_forward_api(None, app, controller)
    update_diffusion_transformer_api(None, app, controller)
    update_edition_api(None, app, controller)
    
    # not close the python
    while True:
        time.sleep(5)