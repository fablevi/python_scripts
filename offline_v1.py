from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image 
import torch
#import transformers
#import jax

#pipeline = StableDiffusionXLPipeline.from_single_file(
#    "../sdxl/sd_xl_base_1.0.safetensors",
#    torch_dtype=torch.float16, 
#    variant="fp16", 
#    use_safetensors=True,
#    local_files_only=True,
#    original_config_file="../sdxl/v2-inference.yaml",
#    load_safety_checker=False

#).to("cuda")
#pipeline_text2image = StableDiffusionXLPipeline.from_pretrained(
#    "../sdxl",
#    torch_dtype=torch.float32,
#    variant="fp16",
#    local_files_only=True,
#    use_safetensors=True
#)

#prompt = "Man drinking coffee"
#image = pipeline_text2image(prompt=prompt).images[0]
#image;

#from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker #NEWLY ADDED safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"), #MODIFIED

pipeline = StableDiffusionXLPipeline.from_single_file(
    "../sdxl/sd_xl_base_1.0.safetensors",
     local_files_only=True,
)
pipeline.to("cpu")

prompt = "a photo of a knight riding a horse on icy lake"
image = pipeline(prompt).images[0]
image.save("result.png")
print("result.png created")