#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# init-agent.sh - Install amct_llm quant Agent Skills into Claude Code and/or OpenCode.
#
# Source of truth: .agents/
#     .agents/skills/       - skill directories
#     .agents/agents/       - subagent markdown files
#     .agents/hooks/        - Claude Code hook scripts
#     .agents/settings.json - Claude Code settings (permissions + hook registration)
#
# Generated (gitignored):
#     .claude/{skills,agents,hooks,docs,settings.json} - Claude Code view
#     .opencode/{skills,agents,docs}                    - OpenCode view
#     ({claude,opencode}/docs symlink .agents/docs so skill→doc relative links resolve in both views)
#     CLAUDE.md -> AGENTS.md                       - Claude Code entry
#
# Usage:
#     bash scripts/init-agent.sh              # both platforms (default)
#     bash scripts/init-agent.sh --claude     # Claude Code only
#     bash scripts/init-agent.sh --opencode   # OpenCode only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$REPO_ROOT/.agents"

if [ ! -d "$SRC" ]; then
    echo "Error: source dir $SRC not found." >&2
    exit 1
fi

arg="${1:-}"
case "$arg" in
    ""|--both|--all) target="both" ;;
    --claude)        target="claude" ;;
    --opencode)      target="opencode" ;;
    -h|--help)
        sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
        exit 0 ;;
    *)
        echo "Unknown argument: $arg" >&2
        echo "Usage: $0 [--claude|--opencode]" >&2
        exit 1 ;;
esac

link_children() {
    # Create child symlinks in $2 for every entry in $1 (absolute paths).
    local src="$1" dst="$2"
    [ -d "$src" ] || return 0
    mkdir -p "$dst"
    # Prune stale symlinks whose source entry no longer exists (renamed/removed skill etc.)
    for existing in "$dst"/*; do
        [ -L "$existing" ] || continue
        [ -e "$src/$(basename "$existing")" ] || { echo "  - $existing pruned (source removed)" >&2; rm -f "$existing"; }
    done
    for entry in "$src"/*; do
        [ -e "$entry" ] || continue
        local target="$dst/$(basename "$entry")"
        if [ -e "$target" ] && [ ! -L "$target" ]; then
            # Non-symlink at a generated-view target: views are fully regenerable,
            # so warn loudly and replace (not silent, no .bak clutter).
            echo "  ! $target was not a symlink; replaced (views are generated)" >&2
            rm -rf "$target"
        fi
        ln -sfn "$(realpath "$entry")" "$target"
    done
}

install_claude() {
    local base="$REPO_ROOT/.claude"
    echo "Installing Claude Code -> $base/"
    link_children "$SRC/skills" "$base/skills"
    link_children "$SRC/agents" "$base/agents"
    link_children "$SRC/hooks"  "$base/hooks"
    [ -d "$SRC/docs" ] && ln -sfn "$(realpath "$SRC/docs")" "$base/docs"
    if [ -f "$SRC/settings.json" ]; then
        cp "$SRC/settings.json" "$base/settings.json"
    fi
    ln -sfn AGENTS.md "$REPO_ROOT/CLAUDE.md"
    echo "  + $base/{skills,agents,hooks,docs,settings.json}"
    echo "  + $REPO_ROOT/CLAUDE.md -> AGENTS.md"
}

install_opencode() {
    local base="$REPO_ROOT/.opencode"
    echo "Installing OpenCode -> $base/"
    link_children "$SRC/skills" "$base/skills"
    link_children "$SRC/agents" "$base/agents"
    [ -d "$SRC/docs" ] && ln -sfn "$(realpath "$SRC/docs")" "$base/docs"
    [ -f "$SRC/opencode.json" ] && cp "$SRC/opencode.json" "$base/opencode.json"
    echo "  + $base/{skills,agents,docs,opencode.json}"
}

case "$target" in
    claude)   install_claude ;;
    opencode) install_opencode ;;
    both)     install_claude; install_opencode ;;
esac

echo "Done."
