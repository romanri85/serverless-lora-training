import os


# Model path key → replacement patterns for each model type.
# {nv} is replaced with the network volume path at runtime.
MODEL_CONFIGS = {
    "flux": {
        "toml_template": "flux.toml",
        "default_epochs": 80,
        "requires_hf_token": True,
        "supports_video": False,
        "path_replacements": {
            "diffusers_path = '/models/flux'": "diffusers_path = '{nv}/models/flux'",
        },
        "validation_paths": [
            "{nv}/models/flux",
        ],
    },
    "sdxl": {
        "toml_template": "sdxl.toml",
        "default_epochs": 100,
        "requires_hf_token": False,
        "supports_video": False,
        "path_replacements": {
            "checkpoint_path = '/models/sdXL_v10VAEFix.safetensors'":
                "checkpoint_path = '{nv}/models/sdXL_v10VAEFix.safetensors'",
        },
        "validation_paths": [
            "{nv}/models/sdXL_v10VAEFix.safetensors",
        ],
    },
    "wan13": {
        "toml_template": "wan13_video.toml",
        "default_epochs": 200,
        "requires_hf_token": False,
        "supports_video": True,
        "path_replacements": {
            "ckpt_path = '/Wan/Wan2.1-T2V-1.3B'":
                "ckpt_path = '{nv}/models/Wan/Wan2.1-T2V-1.3B'",
        },
        "validation_paths": [
            "{nv}/models/Wan/Wan2.1-T2V-1.3B",
        ],
    },
    "wan14b_t2v": {
        "toml_template": "wan14b_t2v.toml",
        "default_epochs": 200,
        "requires_hf_token": False,
        "supports_video": True,
        "path_replacements": {
            "ckpt_path = '/Wan/Wan2.1-T2V-14B'":
                "ckpt_path = '{nv}/models/Wan/Wan2.1-T2V-14B'",
        },
        "validation_paths": [
            "{nv}/models/Wan/Wan2.1-T2V-14B",
        ],
    },
    "wan14b_i2v": {
        "toml_template": "wan14b_i2v.toml",
        "default_epochs": 200,
        "requires_hf_token": False,
        "supports_video": True,
        "path_replacements": {
            "ckpt_path = '/Wan/Wan2.1-I2V-14B-480P'":
                "ckpt_path = '{nv}/models/Wan/Wan2.1-I2V-14B-480P'",
        },
        "validation_paths": [
            "{nv}/models/Wan/Wan2.1-I2V-14B-480P",
        ],
    },
    "qwen": {
        "toml_template": "qwen_toml.toml",
        "default_epochs": 80,
        "requires_hf_token": False,
        "supports_video": False,
        "path_replacements": {
            "diffusers_path = '/models/Qwen-Image'":
                "diffusers_path = '{nv}/models/Qwen-Image'",
        },
        "validation_paths": [
            "{nv}/models/Qwen-Image",
        ],
    },
    "z_image_turbo": {
        "toml_template": "z_image_toml.toml",
        "default_epochs": 80,
        "requires_hf_token": False,
        "supports_video": False,
        "path_replacements": {
            "diffusion_model = '/models/z_image/z_image_turbo_bf16.safetensors'":
                "diffusion_model = '{nv}/models/z_image/z_image_turbo_bf16.safetensors'",
            "vae = '/models/z_image/ae.safetensors'":
                "vae = '{nv}/models/z_image/ae.safetensors'",
            "path = '/models/z_image/qwen_3_4b.safetensors'":
                "path = '{nv}/models/z_image/qwen_3_4b.safetensors'",
            "merge_adapters = ['/models/z_image/zimage_turbo_training_adapter_v2.safetensors']":
                "merge_adapters = ['{nv}/models/z_image/zimage_turbo_training_adapter_v2.safetensors']",
        },
        "validation_paths": [
            "{nv}/models/z_image/z_image_turbo_bf16.safetensors",
            "{nv}/models/z_image/ae.safetensors",
            "{nv}/models/z_image/qwen_3_4b.safetensors",
            "{nv}/models/z_image/zimage_turbo_training_adapter_v2.safetensors",
        ],
    },
    "qwen_image_edit": {
        "toml_template": "qwen_image_edit_toml.toml",
        "default_epochs": 80,
        "requires_hf_token": False,
        "supports_video": False,
        "path_replacements": {
            "diffusers_path = '/models/Qwen-Image'":
                "diffusers_path = '{nv}/models/Qwen-Image'",
            "transformer_path = '/models/qwen_image_edit/qwen_image_edit_2511_bf16.safetensors'":
                "transformer_path = '{nv}/models/qwen_image_edit/qwen_image_edit_2511_bf16.safetensors'",
        },
        "validation_paths": [
            "{nv}/models/Qwen-Image",
            "{nv}/models/qwen_image_edit/qwen_image_edit_2511_bf16.safetensors",
        ],
    },
    "z_image_base": {
        "toml_template": "z_image_base_toml.toml",
        "default_epochs": 80,
        "requires_hf_token": False,
        "supports_video": False,
        "path_replacements": {
            "diffusion_model = '/models/z_image_base/z_image_bf16.safetensors'":
                "diffusion_model = '{nv}/models/z_image_base/z_image_bf16.safetensors'",
            "vae = '/models/z_image/ae.safetensors'":
                "vae = '{nv}/models/z_image/ae.safetensors'",
            "path = '/models/z_image/qwen_3_4b.safetensors'":
                "path = '{nv}/models/z_image/qwen_3_4b.safetensors'",
        },
        "validation_paths": [
            "{nv}/models/z_image_base/z_image_bf16.safetensors",
            "{nv}/models/z_image/ae.safetensors",
            "{nv}/models/z_image/qwen_3_4b.safetensors",
        ],
    },
    "ltx_video": {
        "toml_template": "ltx_video_toml.toml",
        "default_epochs": 80,
        "requires_hf_token": False,
        "supports_video": True,
        "path_replacements": {
            "diffusers_path = '/models/LTX-Video'":
                "diffusers_path = '{nv}/models/LTX-Video'",
        },
        "validation_paths": [
            "{nv}/models/LTX-Video",
        ],
    },
}

# HuggingFace download commands for on-demand model downloads.
# Each entry is a shell command (or chained commands) with {nv} and {token} placeholders.
MODEL_DOWNLOADS = {
    "flux": "huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir {nv}/models/flux --token {token}",
    "sdxl": "huggingface-cli download timoshishi/sdXL_v10VAEFix sdXL_v10VAEFix.safetensors --local-dir {nv}/models/",
    "wan13": "huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir {nv}/models/Wan/Wan2.1-T2V-1.3B",
    "wan14b_t2v": "huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir {nv}/models/Wan/Wan2.1-T2V-14B",
    "wan14b_i2v": "huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir {nv}/models/Wan/Wan2.1-I2V-14B-480P",
    "qwen": "huggingface-cli download Qwen/Qwen-Image --local-dir {nv}/models/Qwen-Image",
    "ltx_video": "huggingface-cli download Lightricks/LTX-Video --local-dir {nv}/models/LTX-Video",
    "z_image_turbo": (
        "mkdir -p {nv}/models/z_image && "
        "huggingface-cli download Comfy-Org/z_image_turbo "
        "split_files/diffusion_models/z_image_turbo_bf16.safetensors "
        "split_files/vae/ae.safetensors "
        "split_files/text_encoders/qwen_3_4b.safetensors "
        "--local-dir /tmp/z_image_turbo_dl && "
        "mv /tmp/z_image_turbo_dl/split_files/diffusion_models/z_image_turbo_bf16.safetensors {nv}/models/z_image/ && "
        "mv /tmp/z_image_turbo_dl/split_files/vae/ae.safetensors {nv}/models/z_image/ && "
        "mv /tmp/z_image_turbo_dl/split_files/text_encoders/qwen_3_4b.safetensors {nv}/models/z_image/ && "
        "rm -rf /tmp/z_image_turbo_dl && "
        "huggingface-cli download ostris/zimage_turbo_training_adapter "
        "zimage_turbo_training_adapter_v2.safetensors "
        "--local-dir {nv}/models/z_image"
    ),
    "z_image_base": (
        "mkdir -p {nv}/models/z_image_base {nv}/models/z_image && "
        "huggingface-cli download Comfy-Org/z_image "
        "split_files/diffusion_models/z_image_bf16.safetensors "
        "--local-dir /tmp/z_image_dl && "
        "mv /tmp/z_image_dl/split_files/diffusion_models/z_image_bf16.safetensors {nv}/models/z_image_base/ && "
        "rm -rf /tmp/z_image_dl && "
        "test -f {nv}/models/z_image/ae.safetensors || ("
        "huggingface-cli download Comfy-Org/z_image_turbo "
        "split_files/vae/ae.safetensors "
        "split_files/text_encoders/qwen_3_4b.safetensors "
        "--local-dir /tmp/z_image_shared_dl && "
        "mv /tmp/z_image_shared_dl/split_files/vae/ae.safetensors {nv}/models/z_image/ && "
        "mv /tmp/z_image_shared_dl/split_files/text_encoders/qwen_3_4b.safetensors {nv}/models/z_image/ && "
        "rm -rf /tmp/z_image_shared_dl)"
    ),
    "qwen_image_edit": (
        "mkdir -p {nv}/models/qwen_image_edit && "
        "huggingface-cli download Comfy-Org/Qwen-Image-Edit_ComfyUI "
        "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors "
        "--local-dir /tmp/qwen_edit_dl && "
        "mv /tmp/qwen_edit_dl/split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors {nv}/models/qwen_image_edit/ && "
        "rm -rf /tmp/qwen_edit_dl && "
        "test -d {nv}/models/Qwen-Image || "
        "huggingface-cli download Qwen/Qwen-Image --local-dir {nv}/models/Qwen-Image"
    ),
}


class ModelRegistry:
    def __init__(self, network_volume):
        self.nv = network_volume

    def get_config(self, model_type):
        if model_type not in MODEL_CONFIGS:
            valid = ", ".join(MODEL_CONFIGS.keys())
            raise ValueError(f"Unknown model_type '{model_type}'. Valid types: {valid}")
        return MODEL_CONFIGS[model_type]

    def validate_model_files(self, model_type):
        config = self.get_config(model_type)
        missing = []
        for path_template in config["validation_paths"]:
            path = path_template.replace("{nv}", self.nv)
            if not os.path.exists(path):
                missing.append(path)
        if missing:
            raise FileNotFoundError(
                f"Model files missing for '{model_type}'. Run setup_network_volume.sh first.\n"
                f"Missing: {missing}"
            )
