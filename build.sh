#!/bin/bash
set -e

# ----------------------------
# Tonkintelligent Build Script
# ----------------------------
ROOT="$(pwd)"
CODE_DIR="$ROOT/code"

ENV_NAME="tonkintelligent"

# Source conda to enable conda activate in script
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]
then
    echo "Activating Conda environment '$ENV_NAME'..."
    conda activate $ENV_NAME
fi

echo "Building Tonkintelligent project data..."
python3 "$CODE_DIR/build.py"

echo "Build completed successfully!"
