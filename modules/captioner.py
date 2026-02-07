import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def caption_images(image_dir, trigger_word=None):
    """
    Caption images using JoyCaption (llama-joycaption-beta-one-hf-llava).

    Creates .txt caption files alongside each image.
    Requires transformers and torch (already in the container).
    """
    image_files = _get_files_by_ext(image_dir, IMAGE_EXTENSIONS)
    if not image_files:
        logger.info("No images to caption")
        return

    # Count how many already have captions
    uncaptioned = [f for f in image_files if not (f.parent / f"{f.stem}.txt").exists()]
    if not uncaptioned:
        logger.info("All images already have captions, skipping")
        return

    logger.info(f"Captioning {len(uncaptioned)} images with JoyCaption...")

    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        from PIL import Image
        import torch

        model_name = "fancyfeast/llama-joycaption-beta-one-hf-llava"
        logger.info(f"Loading JoyCaption model: {model_name}")

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="bfloat16",
            device_map=0,
            trust_remote_code=True,
        )

        tok = processor.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            model.config.pad_token_id = tok.eos_token_id

        prompt = "Write a detailed description for this image in 50 words or less. Do NOT mention any text that is in the image."

        convo = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

        for i, img_path in enumerate(uncaptioned, 1):
            try:
                logger.info(f"[{i}/{len(uncaptioned)}] Captioning {img_path.name}")
                img = Image.open(img_path).convert("RGB")

                inputs = processor(text=[convo_string], images=[img], return_tensors="pt")
                inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                        pad_token_id=tok.pad_token_id,
                        use_cache=True,
                    )

                input_len = inputs["input_ids"].shape[1]
                caption = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

                if trigger_word:
                    caption = f"{trigger_word}, {caption}"

                caption_path = img_path.parent / f"{img_path.stem}.txt"
                caption_path.write_text(caption)
                logger.info(f"Caption: {caption[:80]}...")
            except Exception as e:
                logger.error(f"Failed to caption {img_path.name}: {e}")

        # Free GPU memory
        del model, processor
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("JoyCaption model unloaded")

    except ImportError as e:
        logger.error(f"Missing dependency for image captioning: {e}")
        raise


def caption_videos(video_dir, gemini_api_key, trigger_word=None):
    """
    Caption videos using Gemini API via the TripleX captioner.

    Requires GEMINI_API_KEY and the TripleX repo.
    """
    video_files = _get_files_by_ext(video_dir, VIDEO_EXTENSIONS)
    if not video_files:
        logger.info("No videos to caption")
        return

    uncaptioned = [f for f in video_files if not (f.parent / f"{f.stem}.txt").exists()]
    if not uncaptioned:
        logger.info("All videos already have captions, skipping")
        return

    logger.info(f"Captioning {len(uncaptioned)} videos with Gemini...")

    # Clone TripleX if not present
    triplex_dir = "/TripleX"
    if not os.path.isdir(triplex_dir):
        logger.info("Cloning TripleX captioner...")
        subprocess.run(
            ["git", "clone", "https://github.com/Hearmeman24/TripleX.git", triplex_dir],
            check=True, capture_output=True,
        )

    # Install requirements
    req_path = os.path.join(triplex_dir, "requirements.txt")
    if os.path.exists(req_path):
        subprocess.run(
            ["pip", "install", "-r", req_path],
            capture_output=True,
        )

    # Run gemini captioner
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = gemini_api_key

    result = subprocess.run(
        ["python", os.path.join(triplex_dir, "captioners", "gemini.py"),
         "--dir", str(video_dir), "--max_frames", "1"],
        env=env, capture_output=True, text=True,
    )

    if result.returncode != 0:
        logger.error(f"Video captioning failed: {result.stderr[:500]}")
        raise RuntimeError("Video captioning with Gemini failed")

    # Add trigger word to generated captions if specified
    if trigger_word:
        for vf in uncaptioned:
            caption_path = vf.parent / f"{vf.stem}.txt"
            if caption_path.exists():
                text = caption_path.read_text().strip()
                caption_path.write_text(f"{trigger_word}, {text}")

    logger.info("Video captioning complete")


def _get_files_by_ext(directory, extensions):
    """Get all files with given extensions in a directory."""
    path = Path(directory)
    if not path.exists():
        return []
    return sorted(
        f for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )
