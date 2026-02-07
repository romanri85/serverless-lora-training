#!/bin/bash
# One-time setup: Download all models to the RunPod network volume.
# Run this on an interactive pod with the network volume mounted at /runpod-volume.
#
# Usage: bash setup_network_volume.sh [model_type]
# If no model_type given, downloads ALL models.

set -e

NV="${NETWORK_VOLUME:-/runpod-volume}"
HF_TOKEN="${HF_TOKEN:-}"

echo "================================================"
echo "Network Volume Model Setup"
echo "================================================"
echo "Network volume: $NV"
echo ""

mkdir -p "$NV/models"

download_flux() {
    echo "Downloading FLUX.1-dev (requires HF token)..."
    if [ -z "$HF_TOKEN" ]; then
        echo "ERROR: HF_TOKEN required for Flux. Export HF_TOKEN and retry."
        return 1
    fi
    huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir "$NV/models/flux" --token "$HF_TOKEN"
    echo "✅ Flux downloaded"
}

download_sdxl() {
    echo "Downloading SDXL v1.0 VAE Fix..."
    huggingface-cli download timoshishi/sdXL_v10VAEFix sdXL_v10VAEFix.safetensors --local-dir "$NV/models/"
    echo "✅ SDXL downloaded"
}

download_wan13() {
    echo "Downloading Wan2.1-T2V-1.3B..."
    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$NV/models/Wan/Wan2.1-T2V-1.3B"
    echo "✅ Wan 1.3B downloaded"
}

download_wan14b_t2v() {
    echo "Downloading Wan2.1-T2V-14B..."
    huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir "$NV/models/Wan/Wan2.1-T2V-14B"
    echo "✅ Wan 14B T2V downloaded"
}

download_wan14b_i2v() {
    echo "Downloading Wan2.1-I2V-14B-480P..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$NV/models/Wan/Wan2.1-I2V-14B-480P"
    echo "✅ Wan 14B I2V downloaded"
}

download_qwen() {
    echo "Downloading Qwen-Image..."
    huggingface-cli download Qwen/Qwen-Image --local-dir "$NV/models/Qwen-Image"
    echo "✅ Qwen-Image downloaded"
}

download_ltx_video() {
    echo "Downloading LTX-Video..."
    huggingface-cli download Lightricks/LTX-Video --local-dir "$NV/models/LTX-Video"
    echo "✅ LTX-Video downloaded"
}

# z_image_turbo, z_image_base, and qwen_image_edit models are not on public HF repos.
# Add download commands here if/when they become available, or copy manually.

if [ -n "$1" ]; then
    case "$1" in
        flux)         download_flux ;;
        sdxl)         download_sdxl ;;
        wan13)        download_wan13 ;;
        wan14b_t2v)   download_wan14b_t2v ;;
        wan14b_i2v)   download_wan14b_i2v ;;
        qwen)         download_qwen ;;
        ltx_video)    download_ltx_video ;;
        *)
            echo "Unknown model: $1"
            echo "Valid: flux, sdxl, wan13, wan14b_t2v, wan14b_i2v, qwen, ltx_video"
            exit 1
            ;;
    esac
else
    echo "Downloading ALL public models..."
    echo ""
    download_flux || true
    download_sdxl
    download_wan13
    download_wan14b_t2v
    download_wan14b_i2v
    download_qwen
    download_ltx_video
fi

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo "Models directory: $NV/models/"
ls -la "$NV/models/"
