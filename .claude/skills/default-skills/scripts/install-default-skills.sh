#!/bin/bash

# 脚本功能：安装项目必备的 skills
# 默认技能技能列表
DEFAULT_SKILLS=("gitcode-pr" "gitcode-issue")

# 脚本所在目录的上上级目录为 .claude/skills/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILLS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 创建临时目录
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# 克隆 skills 仓库
echo "Cloning skills repository..."
git clone --depth 1 https://gitcode.com/cann-agent/skills.git "$TEMP_DIR/skills"
if [ $? -ne 0 ]; then
    echo "Error: Failed to clone skills repository"
    exit 1
fi

# 检查 skills 目录是否存在
if [ ! -d "$TEMP_DIR/skills/skills" ]; then
    echo "Error: skills directory not found in repository"
    exit 1
fi

# 拷贝技能到 .claude/skills/ 目录
echo "Installing skills..."
for skill in "${DEFAULT_SKILLS[@]}"; do
    if [ -d "$TEMP_DIR/skills/skills/$skill" ]; then
        cp -r "$TEMP_DIR/skills/skills/$skill" "$SKILLS_DIR/"
        echo "Installed skill: $skill"
    else
        echo "Warning: Skill '$skill' not found in repository"
    fi
done

echo "All skills installed successfully."
