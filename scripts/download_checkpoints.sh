#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Define URLs and target paths
CHECKPOINT_URL="https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth"
CHECKPOINT_TARGET="checkpoints/sam_hq_vit_l.pth"

# Create directories if they don't exist
mkdir -p checkpoints
mkdir -p "$CONFIG_TARGET_DIR"

# Download checkpoint if it doesn't exist
if [ ! -f "$CHECKPOINT_TARGET" ]; then
  echo "Downloading $CHECKPOINT_TARGET..."
  wget -O "$CHECKPOINT_TARGET" "$CHECKPOINT_URL"
else
  echo "$CHECKPOINT_TARGET already exists."
fi

echo "Checkpoint download process finished."
