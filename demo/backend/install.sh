#!/bin/bash
set -e

echo "Setting up Backend..."

# Ensure SAM2 repo exists
if [ ! -d "../../sam2" ]; then
    echo "Cloning SAM2..."
    git clone https://github.com/facebookresearch/sam2.git ../../sam2
fi

# Upgrade pip
pip install --upgrade pip

# Install backend requirements
pip install -r requirements.txt

# Install SAM2 (editable mode)
pip install -e ../../sam2

echo "Backend setup complete."
