#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")"

if command -v uv &>/dev/null; then
    echo "Using uv"

    if [[ ! -d .venv ]]; then
        echo "Creating virtual environment (Python 3.13)..."
        uv venv -p 3.13
    fi

    source .venv/bin/activate

    echo "Installing torch..."
    uv pip install torch torchvision --torch-backend auto

    echo "Installing requirements..."
    uv pip install -r requirements.txt
else
    echo "uv not found, falling back to pip"
    echo "  (recommended: install uv first with ./install-uv.sh)"

    if [[ ! -d .venv ]]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
    fi

    source .venv/bin/activate
    pip install -U pip setuptools wheel
    pip install -r requirements.txt
fi

echo ""
echo "Setup complete. Run 'source .venv/bin/activate' to enter the virtual environment."
