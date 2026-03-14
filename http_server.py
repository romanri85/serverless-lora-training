"""
FastAPI server for LoRA training.

Exposes /health and /train endpoints. Wraps handler.py which supports
all 15 model types (flux, flux2, sdxl, wan13, wan14b_t2v, wan14b_i2v,
wan22_t2v_high, wan22_t2v_low, wan22_i2v_high, wan22_i2v_low,
qwen, z_image_turbo, qwen_image_edit, z_image_base, ltx_video).
"""

import logging
import os
import subprocess
import sys
import uuid

from fastapi import FastAPI, Request
import uvicorn

from handler import handler as original_handler
from modules.model_registry import ModelRegistry, MODEL_DOWNLOADS

app = FastAPI()
logger = logging.getLogger(__name__)

NETWORK_VOLUME = os.environ.get("NETWORK_VOLUME", "/workspace/models")


def download_model_if_needed(model_type: str, hf_token: str = None):
    """Download model files on-demand if not already present."""
    registry = ModelRegistry(NETWORK_VOLUME)
    try:
        registry.validate_model_files(model_type)
        logger.info(f"Model '{model_type}' files already present")
    except FileNotFoundError:
        if model_type not in MODEL_DOWNLOADS:
            logger.warning(
                f"No auto-download available for '{model_type}'. "
                "Model files must be pre-loaded on the volume."
            )
            return
        logger.info(f"Model '{model_type}' not found, downloading...")
        cmd = MODEL_DOWNLOADS[model_type].replace("{nv}", NETWORK_VOLUME)
        if hf_token:
            cmd = cmd.replace("{token}", hf_token)
        os.makedirs(f"{NETWORK_VOLUME}/models", exist_ok=True)
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"Model '{model_type}' download complete")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/train")
async def train(request: Request):
    body = await request.json()

    if body.get("dry_run"):
        return {"status": "benchmark_ok"}

    # Download model on-demand before training
    model_type = body.get("model_type")
    hf_token = body.get("hf_token")
    if model_type:
        download_model_if_needed(model_type, hf_token)

    job = {"id": str(uuid.uuid4()), "input": body}

    # Consume the generator handler, collect progress updates
    last_result = None
    for update in original_handler(job):
        logger.info(f"Progress: {update}")
        last_result = update

    return last_result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger.info("Application startup complete")
    uvicorn.run(app, host="0.0.0.0", port=18000)
