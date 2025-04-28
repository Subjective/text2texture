#!/bin/bash
set -e

echo "Setting up Backend..."

# Upgrade pip
pip install --upgrade pip

# Install backend requirements
pip install -r requirements.txt

echo "Backend setup complete."
