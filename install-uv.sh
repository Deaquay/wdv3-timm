#!/usr/bin/env bash
set -euo pipefail

echo "Installing uv..."

if command -v curl &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
elif command -v wget &>/dev/null; then
    wget -qO- https://astral.sh/uv/install.sh | sh
else
    echo "Error: curl or wget is required to install uv."
    exit 1
fi

echo "uv installed. You may need to restart your shell or source your profile."
