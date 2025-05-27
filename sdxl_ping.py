import os
import replicate
import sys
import logging

# \u25BC ここに実際の Replicate トークンを入れる
os.environ["REPLICATE_API_TOKEN"] = "r8_********************************"

MODEL = "replicate-sdxl"

prompt = "red and blue test"
params = {"prompt": prompt}

logger = logging.getLogger(__name__)

logger.debug("Replicate prompt: %s", prompt)
logger.debug("Replicate request params: %s", params)
try:
    url = replicate.Client().run(MODEL, input=params)
    logger.debug("Replicate raw response: %s", url)
    logger.info("\u2705 \u751f\u6210URL: %s", url)
except Exception as e:
    logger.error("\u274c ERROR: %s - %s", e.__class__.__name__, e)
    sys.exit(1)
