import type { Plugin } from "@opencode-ai/plugin"
import * as fs from "fs"
import * as path from "path"
import crypto from "crypto"

const logFile = path.join(__dirname, "install_error.log")

function log(message: string) {
  const timestamp = new Date().toISOString()
  fs.appendFileSync(logFile, `[${timestamp}] ${message}\n`)
}

function getFileHash(filePath: string): string | null {
  if (!fs.existsSync(filePath)) return null
  try {
    const content = fs.readFileSync(filePath, 'utf-8')
    return crypto.createHash('md5').update(content).digest('hex')
  } catch (error) {
    return null
  }
}

interface SkillState {
  name: string
  exists: boolean
  hash: string | null
}

function findGitRoot(startDir: string): string {
  let dir = startDir
  while (dir !== path.dirname(dir)) {
    if (fs.existsSync(path.join(dir, ".git"))) {
      return dir
    }
    dir = path.dirname(dir)
  }
  return startDir
}

export const InstallSkillsPlugin: Plugin = async ({ $, directory }) => {
  const rootDir = findGitRoot(directory)
  const installSkills = async () => {
    try {
      // 记录安装前两个skill文件的状态
      const skillsToCheck = [
        { name: 'gitcode-pr', path: path.join(rootDir, ".claude", "skills", "gitcode-pr", "SKILL.md") },
        { name: 'gitcode-issue', path: path.join(rootDir, ".claude", "skills", "gitcode-issue", "SKILL.md") }
      ]

      const beforeStates: SkillState[] = skillsToCheck.map(skill => ({
        name: skill.name,
        exists: fs.existsSync(skill.path),
        hash: getFileHash(skill.path)
      }))
      // 检测是否为Windows系统
      const isWindows = process.platform === 'win32'
      if (isWindows) {
        process.stdout.write(`💡 提示：如果需要安装或更新默认skill，请输入指令"安装默认skill"\n\n`)
        return
      }
      const scriptPath = path.join(rootDir, ".claude", "skills", "default-skills", "scripts", "install-default-skills.sh")
      await $`bash ${scriptPath} > /dev/null`

      // 记录安装后两个skill文件的状态
      const afterStates: SkillState[] = skillsToCheck.map(skill => ({
        name: skill.name,
        exists: fs.existsSync(skill.path),
        hash: getFileHash(skill.path)
      }))

      // 检查是否有任何skill发生了变化
      let hasChanges = false
      const changedSkills: string[] = []

      for (let i = 0; i < beforeStates.length; i++) {
        const before = beforeStates[i]
        const after = afterStates[i]

        if (!before.exists && after.exists) {
          hasChanges = true
          changedSkills.push(`${before.name} 新安装`)
        } else if (before.exists && after.exists && before.hash !== after.hash) {
          hasChanges = true
          changedSkills.push(`${before.name} 已更新`)
        }
      }

      // 只有当两个skill都在安装前后完全相同时才不打印提示
      if (hasChanges && changedSkills.length > 0) {
        setTimeout(() => {
        process.stdout.write(`💡 ${changedSkills.join(', ')}，重启opencode才能完全生效\n\n`)
        }, 1000)
      }
    } catch (error) {
      log(`Command failed: ${error.message}`)
      if (error.stderr) log(`stderr from error: ${error.stderr}`)
      // 创建错误标记文件，供用户后续查看
      const errorMarkerPath = path.join(rootDir, ".opencode_skills_error")
      const detail = error.stderr ? `${error.message}\n${error.stderr}` : error.message
      const errorMessage = `❌ 安装默认技能时出错了，请输入指令"安装默认skill"重新安装\n错误详情: ${detail}\n`
      fs.writeFileSync(errorMarkerPath, errorMessage)

      // 延迟打印到标准输出，避免被界面刷新清除
      setTimeout(() => {
        process.stdout.write(`❌ 安装默认技能时出错了，请输入指令"安装默认skill"重新安装\n`)
        process.stdout.write(`   错误详情: ${detail}\n`)
        process.stdout.write(`   错误详情请查看: ${errorMarkerPath}\n\n`)
      }, 2000)
    }
  };

  installSkills();
  return {
    event: async ({ event }) => {}
  }
}
