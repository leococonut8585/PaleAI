import os
import replicate

sd_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

MODEL_ID = "stability-ai/sdxl"
