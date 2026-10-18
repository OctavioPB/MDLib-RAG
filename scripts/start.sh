#!/usr/bin/env sh
# One-step launcher (ADR-021) for macOS and Linux: installs what is missing, then `biblio start`.
set -e
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then
    PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv, the tool that installs Python and this app's packages, is not installed."
    printf "Install it now from astral.sh? [Y/n] "
    read -r answer
    case "$answer" in
        n|N|no|No) echo "Install uv from https://docs.astral.sh/uv/ and run this again."; exit 1 ;;
    esac
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$PATH"
fi

echo "Checking Python and the app's packages (downloads only what is missing)..."
uv sync
exec uv run biblio start "$@"
