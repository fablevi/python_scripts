from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image 
import torch
import transformers

pipeline_text2image = StableDiffusionXLPipeline.from_pretrained(
    "../sdxl",
    from_tf=False
)
pipeline_text2image.to("cpu")
#pipeline_text2image.enable_model_cpu_offload()

prompt = "a photo of a knight riding a horse on icy lake"
image = pipeline_text2image(prompt).images[0]
image.save("result.png")
print("result.png created")

#    torch_dtype=torch.float32,
#    variant="fp16",
#    local_files_only=True,
#    use_safetensors=True