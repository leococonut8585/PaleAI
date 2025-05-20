import os, replicate, sys

# ▼ ここに実際の Replicate トークンを入れる
os.environ["REPLICATE_API_TOKEN"] = "r8_********************************"

MODEL = "stability-ai/sdxl@7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc"

try:
    url = replicate.Client().run(MODEL, input={"prompt": "red and blue test"})
    print("✅ 生成URL:", url)
except Exception as e:
    print("❌ ERROR:", e.__class__.__name__, "-", e)
    sys.exit(1)
