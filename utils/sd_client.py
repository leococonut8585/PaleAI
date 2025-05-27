import os
import replicate

sd_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

# Use the latest SDXL model on Replicate for all image generation
MODEL_ID = "replicate-sdxl"
