import os
import replicate

sd_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

# Use the latest SDXL model on Replicate for all image generation
MODEL_ID = (
    "stability-ai/sdxl@7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc"
)
