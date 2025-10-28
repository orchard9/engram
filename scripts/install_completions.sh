#!/bin/bash
# Install shell completion scripts for engram CLI
#
# Usage:
#   ./scripts/install_completions.sh [shell]
#
# Arguments:
#   shell: bash, zsh, fish, or detect (default: detect)
#
# This script installs the engram shell completion scripts to the appropriate
# system locations based on the detected or specified shell.

set -euo pipefail

SHELL_TYPE="${1:-detect}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPLETIONS_DIR="$PROJECT_ROOT/completions"

# Colors for output (if supported)
if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    RED=$(tput setaf 1)
    RESET=$(tput sgr0)
else
    GREEN=""
    YELLOW=""
    RED=""
    RESET=""
fi

detect_shell() {
    case "$SHELL" in
        */bash) echo "bash" ;;
        */zsh) echo "zsh" ;;
        */fish) echo "fish" ;;
        *) echo "unknown" ;;
    esac
}

if [ "$SHELL_TYPE" = "detect" ]; then
    SHELL_TYPE=$(detect_shell)
    echo "${YELLOW}Auto-detected shell: ${SHELL_TYPE}${RESET}"
fi

# Check if completions directory exists
if [ ! -d "$COMPLETIONS_DIR" ]; then
    echo "${RED}Error: Completions directory not found: $COMPLETIONS_DIR${RESET}"
    echo "Run 'cargo build' first to generate completion scripts"
    exit 1
fi

case "$SHELL_TYPE" in
    bash)
        # Bash completion directory
        if [ -d "$HOME/.local/share/bash-completion/completions" ]; then
            COMPLETION_DIR="$HOME/.local/share/bash-completion/completions"
        elif [ -d "$HOME/.bash_completion.d" ]; then
            COMPLETION_DIR="$HOME/.bash_completion.d"
        else
            COMPLETION_DIR="$HOME/.local/share/bash-completion/completions"
            mkdir -p "$COMPLETION_DIR"
        fi

        if [ ! -f "$COMPLETIONS_DIR/engram.bash" ]; then
            echo "${RED}Error: engram.bash not found in $COMPLETIONS_DIR${RESET}"
            exit 1
        fi

        cp "$COMPLETIONS_DIR/engram.bash" "$COMPLETION_DIR/engram"
        echo "${GREEN}Bash completion installed to: $COMPLETION_DIR/engram${RESET}"
        echo ""
        echo "To enable completions in your current shell:"
        echo "  ${YELLOW}source $COMPLETION_DIR/engram${RESET}"
        echo ""
        echo "Completions will be automatically loaded in new shells"
        ;;

    zsh)
        # Zsh completion directory
        if [ -n "${ZDOTDIR:-}" ]; then
            COMPLETION_DIR="${ZDOTDIR}/.zfunc"
        elif [ -d "$HOME/.zsh/completion" ]; then
            COMPLETION_DIR="$HOME/.zsh/completion"
        else
            COMPLETION_DIR="$HOME/.zfunc"
        fi
        mkdir -p "$COMPLETION_DIR"

        if [ ! -f "$COMPLETIONS_DIR/_engram" ]; then
            echo "${RED}Error: _engram not found in $COMPLETIONS_DIR${RESET}"
            exit 1
        fi

        cp "$COMPLETIONS_DIR/_engram" "$COMPLETION_DIR/_engram"
        echo "${GREEN}Zsh completion installed to: $COMPLETION_DIR/_engram${RESET}"
        echo ""
        echo "To enable completions, add the following to your ~/.zshrc:"
        echo "  ${YELLOW}fpath=($COMPLETION_DIR \$fpath)${RESET}"
        echo "  ${YELLOW}autoload -U compinit && compinit${RESET}"
        echo ""
        echo "Then restart your shell or run:"
        echo "  ${YELLOW}source ~/.zshrc${RESET}"
        ;;

    fish)
        # Fish completion directory
        COMPLETION_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/fish/completions"
        mkdir -p "$COMPLETION_DIR"

        if [ ! -f "$COMPLETIONS_DIR/engram.fish" ]; then
            echo "${RED}Error: engram.fish not found in $COMPLETIONS_DIR${RESET}"
            exit 1
        fi

        cp "$COMPLETIONS_DIR/engram.fish" "$COMPLETION_DIR/engram.fish"
        echo "${GREEN}Fish completion installed to: $COMPLETION_DIR/engram.fish${RESET}"
        echo ""
        echo "Completions will be automatically loaded in new shells"
        ;;

    unknown)
        echo "${RED}Error: Could not detect shell type${RESET}"
        echo "Your current SHELL: $SHELL"
        echo ""
        echo "Specify shell explicitly:"
        echo "  ${YELLOW}$0 bash${RESET}"
        echo "  ${YELLOW}$0 zsh${RESET}"
        echo "  ${YELLOW}$0 fish${RESET}"
        exit 1
        ;;

    *)
        echo "${RED}Error: Unknown or unsupported shell: $SHELL_TYPE${RESET}"
        echo "Supported shells: bash, zsh, fish"
        exit 1
        ;;
esac

echo ""
echo "${GREEN}Installation complete!${RESET}"
echo "Test completions by typing: ${YELLOW}engram <TAB>${RESET}"
