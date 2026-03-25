import modal
app =modal.App("comfyui-environment")

  USER_AGENT ="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0"
cimage  = (modal.Image.debian_slim(python_version="3.11")
    
    .apt_install("git", "wget", "aria2", "libgl1-mesa-glx", "libglib2.0-0")
    
  
    .pip_install("torch", "torchvision", "torchaudio", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("huggingface_hub[cli,hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}) 
    
   
    .run_commands("git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI")
    .run_commands("cd /workspace/ComfyUI && pip install -r requirements.txt")


    .run_commands("git clone https://github.com/evanspearman/ComfyMath /workspace/ComfyUI/custom_nodes/ComfyMath")

   
    .run_commands(f"python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Lightricks/LTX-2.3-fp8', filename='ltx-2.3-22b-dev-fp8.safetensors', local_dir='/workspace/ComfyUI/models/checkpoints', user_agent='{USER_AGENT}')\"")
    
    
    .run_commands(f"python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Lightricks/LTX-2.3', filename='ltx-2.3-22b-distilled-lora-384.safetensors', local_dir='/workspace/ComfyUI/models/loras', user_agent='{USER_AGENT}')\"")
    
   
    .run_commands(f"python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Lightricks/LTX-2.3', filename='ltx-2.3-spatial-upscaler-x2-1.1.safetensors', local_dir='/workspace/ComfyUI/models/upscale_models', user_agent='{USER_AGENT}')\"")


    .run_commands(f"wget -U \"{USER_AGENT}\" -O /workspace/ComfyUI/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors")


    .run_commands(f"python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ByteDance/SDXL-Lightning', filename='sdxl_lightning_4step.safetensors', local_dir='/workspace/ComfyUI/models/checkpoints', user_agent='{USER_AGENT}')\"")
)

@app.function(image = "cimage" , gpu = "H100" , timeout = 3600)
@modal.web_server(port = "8000" , startup_timeout = 100)
def start_comfyui():
  import subprocess
  subprocess.run(["python", "/workspace/ComfyUI/main.py", "--listen", "0.0.0.0", "--port", "8188"])
