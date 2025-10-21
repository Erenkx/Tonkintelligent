#!/bin/bash
set -e

# -----------------------------------
# Tonkintelligent Installation Script
# -----------------------------------
ENV_NAME="tonkintelligent"
PYTHON_VERSION="3.10"
REQ_FILE="requirements.txt"
TOS_CHANNELS=(
    "https://repo.anaconda.com/pkgs/main"
    "https://repo.anaconda.com/pkgs/r"
)

ensure_conda_tos_acceptance() {
    local channel

    for channel in "${TOS_CHANNELS[@]}"; do
        echo "Ensuring Terms of Service accepted for $channel..."
        if conda tos accept --override-channels --channel "$channel" >/dev/null 2>&1; then
            echo "Terms accepted for $channel"
            continue
        fi

        echo "Retrying ToS acceptance for $channel with automatic confirmation..."
        if yes | conda tos accept --override-channels --channel "$channel" >/dev/null 2>&1; then
            echo "Terms accepted for $channel"
        else
            echo "Failed to accept Terms of Service for $channel"
            exit 1
        fi
    done
}

echo "Starting installation for Tonkintelligent..."

# ------------------------
# Check Conda Installation
# ------------------------
if ! command -v conda &> /dev/null
then
    echo "Conda not found. Installing Miniconda..."

    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
    bash Miniconda3-latest-MacOSX-arm64.sh -b
    rm Miniconda3-latest-MacOSX-arm64.sh
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda init zsh
    echo "Miniconda installed successfully"
else
    echo "Conda found: $(conda --version)"
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

ensure_conda_tos_acceptance # Comment this line out if you have already accepted the ToS on the Conda website manually

# ------------------------
# Create Conda Environment
# ------------------------
if conda env list | grep -q "$ENV_NAME"
then
    echo "Environment '$ENV_NAME' already exists"
else
    echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
    echo "Environment '$ENV_NAME' created successfully"
fi

# --------------------------
# Activate Conda Environment
# --------------------------
echo "Activating environment..."
conda activate $ENV_NAME

# -------------------------
# Requirements Installation
# -------------------------
if [ -f "$REQ_FILE" ]
then
    echo "Installing Python dependencies from $REQ_FILE..."
    pip install --upgrade pip
    pip install -r $REQ_FILE
    echo "Dependencies installed successfully!"
else
    echo "Requirements file '$REQ_FILE' not found!"
    exit 1
fi

echo "Installation completed successfully! To activate the environment later, run:"
echo "      conda activate $ENV_NAME"
