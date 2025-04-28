#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Define URLs and target paths
CHECKPOINT_URL1="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
CHECKPOINT_TARGET1="checkpoints/sam2.1_hiera_large.pt"

CHECKPOINT_URL2="https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth" # Updated URL for direct download
CHECKPOINT_TARGET2="checkpoints/sam_hq_vit_l.pth"

CONFIG_URL="https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
CONFIG_TARGET_DIR="configs/sam2.1"
CONFIG_TARGET_FILE="$CONFIG_TARGET_DIR/sam2.1_hiera_l.yaml"

# Create directories if they don't exist
mkdir -p checkpoints
mkdir -p "$CONFIG_TARGET_DIR"

# Download checkpoint 1 if it doesn't exist
if [ ! -f "$CHECKPOINT_TARGET1" ]; then
  echo "Downloading $CHECKPOINT_TARGET1..."
  wget -O "$CHECKPOINT_TARGET1" "$CHECKPOINT_URL1"
else
  echo "$CHECKPOINT_TARGET1 already exists."
fi

# Download checkpoint 2 if it doesn't exist
if [ ! -f "$CHECKPOINT_TARGET2" ]; then
  echo "Downloading $CHECKPOINT_TARGET2..."
  wget -O "$CHECKPOINT_TARGET2" "$CHECKPOINT_URL2"
else
  echo "$CHECKPOINT_TARGET2 already exists."
fi

# Download config file if it doesn't exist
if [ ! -f "$CONFIG_TARGET_FILE" ]; then
  echo "Downloading $CONFIG_TARGET_FILE..."
  wget -O "$CONFIG_TARGET_FILE" "$CONFIG_URL"
else
  echo "$CONFIG_TARGET_FILE already exists."
fi

echo "Checkpoint and config download process finished."
