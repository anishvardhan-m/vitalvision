#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH="$MODELS_DIR/face_landmarker.task"

mkdir -p "$MODELS_DIR"
curl -L -o "$MODEL_PATH" "$MODEL_URL"
echo "Downloaded to $MODEL_PATH"
