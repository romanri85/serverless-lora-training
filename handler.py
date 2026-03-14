"""
LoRA Training Handler.

Accepts a job payload, runs the full pipeline:
  Download HF dataset → Caption (optional) → Train LoRA → Upload to HF

Returns the HuggingFace repo URL with the trained LoRA.
"""

import os
import signal
import logging
import traceback

import toml

from modules.model_registry import ModelRegistry
from modules.flash_attn_installer import ensure_flash_attn
from modules.dataset_manager import download_dataset
from modules.captioner import caption_images, caption_videos
from modules.config_builder import build_configs
from modules.trainer import TrainingRunner, find_all_checkpoints
from modules.uploader import upload_all_checkpoints
from modules.s3_uploader import upload_checkpoint_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Network volume mount point
NETWORK_VOLUME = os.environ.get("NETWORK_VOLUME", "/workspace/models")

# Active trainer for SIGTERM handling
_active_trainer = None


def _sigterm_handler(signum, frame):
    logger.info("Received SIGTERM, shutting down gracefully...")
    if _active_trainer:
        _active_trainer.terminate()


signal.signal(signal.SIGTERM, _sigterm_handler)


def handler(job):
    """
    Generator handler for LoRA training.

    Yields progress updates, returns final result with HF repo URL.
    """
    global _active_trainer
    job_input = job["input"]
    job_id = job.get("id", "unknown")

    # Work directory for this job
    work_dir = f"/tmp/job_{job_id}"
    os.makedirs(work_dir, exist_ok=True)

    try:
        # ── Extract params ──────────────────────────────────────
        model_type = job_input["model_type"]
        hf_dataset_repo = job_input["hf_dataset_repo"]
        hf_output_repo = job_input.get("hf_output_repo")
        hf_token = job_input.get("hf_token")

        caption_mode = job_input.get("caption_mode")  # "images", "videos", "both", None
        trigger_word = job_input.get("trigger_word")
        gemini_api_key = job_input.get("gemini_api_key")
        training_params = job_input.get("training_params", {})
        private_repo = job_input.get("private", False)
        num_gpus = job_input.get("num_gpus", 1)

        # ── Phase 1: Validation (0%) ────────────────────────────
        yield {"progress": 0, "message": f"Validating job for model '{model_type}'..."}

        registry = ModelRegistry(NETWORK_VOLUME)
        config = registry.get_config(model_type)

        if config["requires_hf_token"] and not hf_token:
            yield {"error": f"Model '{model_type}' requires an HF token"}
            return

        # Validate model files exist on network volume
        registry.validate_model_files(model_type)
        yield {"progress": 1, "message": "Model files validated"}

        # ── Phase 2: flash-attn (1-5%) ──────────────────────────
        yield {"progress": 2, "message": "Ensuring flash-attn is installed..."}
        ensure_flash_attn()
        yield {"progress": 5, "message": "flash-attn ready"}

        # ── Phase 3: Dataset download (5-10%) ───────────────────
        yield {"progress": 5, "message": f"Downloading dataset from {hf_dataset_repo}..."}
        dataset_info = download_dataset(hf_dataset_repo, hf_token, work_dir)
        img_count = dataset_info.get("image_count", 0)
        vid_count = dataset_info.get("video_count", 0)
        yield {
            "progress": 10,
            "message": f"Dataset ready: {img_count} images, {vid_count} videos",
        }

        # ── Phase 4: Captioning (10-25%) ────────────────────────
        if caption_mode in ("images", "both"):
            yield {"progress": 10, "message": "Captioning images with JoyCaption..."}
            caption_images(dataset_info["images"], trigger_word=trigger_word)
            yield {"progress": 18, "message": "Image captioning complete"}

        if caption_mode in ("videos", "both"):
            if not gemini_api_key:
                yield {"error": "gemini_api_key required for video captioning"}
                return
            yield {"progress": 18, "message": "Captioning videos with Gemini..."}
            caption_videos(dataset_info["videos"], gemini_api_key, trigger_word=trigger_word)
            yield {"progress": 25, "message": "Video captioning complete"}

        if not caption_mode:
            yield {"progress": 25, "message": "Skipping captioning (captions expected in dataset)"}

        # ── Phase 5: Config build (25-30%) ──────────────────────
        yield {"progress": 25, "message": "Building training configuration..."}
        configs = build_configs(
            model_type=model_type,
            network_volume=NETWORK_VOLUME,
            work_dir=work_dir,
            dataset_info=dataset_info,
            training_params=training_params,
            num_gpus=num_gpus,
        )
        yield {"progress": 30, "message": "Training config ready"}

        # ── Phase 6: Training (30-90%) ──────────────────────────
        yield {"progress": 30, "message": "Starting LoRA training..."}

        # Read total_epochs from generated config for progress tracking
        with open(configs["training_toml"], "r") as f:
            parsed_config = toml.load(f)
        total_epochs = parsed_config.get("epochs")

        trainer = TrainingRunner(configs["training_toml"], work_dir, total_epochs=total_epochs, num_gpus=num_gpus)
        _active_trainer = trainer

        for update in trainer.run():
            yield update

        _active_trainer = None

        # ── Phase 7: Upload (90-100%) ───────────────────────────
        yield {"progress": 90, "message": "Finding checkpoints..."}

        checkpoints = find_all_checkpoints(configs["output_dir"])
        if not checkpoints:
            yield {"error": "No checkpoint found after training. Training may have failed."}
            return

        yield {"progress": 90, "message": f"Found {len(checkpoints)} checkpoint(s)"}

        output = {}

        # Upload to S3 if configured (DO Spaces or any S3-compatible)
        s3_bucket = os.environ.get("S3_BUCKET_NAME")
        if s3_bucket:
            latest_dir = checkpoints[-1][1]
            yield {"progress": 91, "message": f"Uploading LoRA to S3 ({s3_bucket})..."}
            try:
                s3_files = upload_checkpoint_dir(latest_dir, model_type)
                output["s3_files"] = s3_files
                logger.info(f"S3 upload complete: {len(s3_files)} file(s)")
            except Exception as e:
                logger.error(f"S3 upload failed: {e}")
                yield {"progress": 92, "message": f"S3 upload failed: {e}, continuing..."}

        # Upload all checkpoints to HuggingFace if repo specified
        if hf_output_repo:
            yield {"progress": 93, "message": f"Uploading {len(checkpoints)} checkpoint(s) to {hf_output_repo}..."}

            upload_params = dict(training_params)
            if total_epochs and "epochs" not in upload_params:
                upload_params["epochs"] = total_epochs

            repo_url = upload_all_checkpoints(
                checkpoints=checkpoints,
                hf_output_repo=hf_output_repo,
                hf_token=hf_token,
                model_type=model_type,
                training_params=upload_params,
                private=private_repo,
                trigger_word=trigger_word,
            )
            output["repo_url"] = repo_url

        yield {
            "progress": 100,
            "message": "Done!",
            "output": output,
            "refresh_worker": True,
        }

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        yield {"error": str(e), "refresh_worker": True}
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        yield {"error": str(e), "refresh_worker": True}
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        yield {"error": str(e), "refresh_worker": True}
    except Exception as e:
        logger.error(f"Unexpected error: {traceback.format_exc()}")
        yield {"error": f"Unexpected error: {str(e)}", "refresh_worker": True}
    finally:
        _active_trainer = None

