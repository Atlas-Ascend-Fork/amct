#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
set -e

# 脚本功能：补装或强制更新项目必备的 skills
# 用法：install-default-skills.sh [--force]
#   默认：仅补装缺失的 skill，已存在的跳过
#   --force：从上游强制覆盖所有 skill（用于主动更新）
DEFAULT_SKILLS=("gitcode-pr" "gitcode-issue")

FORCE=false
for arg in "$@"; do
    [ "$arg" = "--force" ] && FORCE=true
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILLS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

echo "Cloning skills repository..."
git clone --depth 1 https://gitcode.com/cann-agent/skills.git "$TEMP_DIR/skills"
if [ $? -ne 0 ]; then
    echo "Error: Failed to clone skills repository"
    exit 1
fi

if [ ! -d "$TEMP_DIR/skills/skills" ]; then
    echo "Error: skills directory not found in repository"
    exit 1
fi

installed=0
skipped=0
for skill in "${DEFAULT_SKILLS[@]}"; do
    if [ -d "$SKILLS_DIR/$skill" ] && [ "$FORCE" = false ]; then
        echo "Skipped (already exists): $skill"
        skipped=$((skipped + 1))
        continue
    fi
    if [ -d "$TEMP_DIR/skills/skills/$skill" ]; then
        rm -rf "$SKILLS_DIR/$skill"
        cp -r "$TEMP_DIR/skills/skills/$skill" "$SKILLS_DIR/"
        echo "Installed skill: $skill"
        installed=$((installed + 1))
    else
        echo "Warning: Skill '$skill' not found in repository"
    fi
done

if [ "$installed" -eq 0 ]; then
    echo "All skills already present, nothing to install. Use --force to update."
else
    echo "$installed skill(s) installed, $skipped skipped."
fi
