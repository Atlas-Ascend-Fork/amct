# PR文档翻译Skill

## 功能描述
检测用户本地仓库的md文档，自动翻译成英文文档。支持：
- 非docs目录下的所有md文档
- docs目录下，排除列表外的md文档

## 触发条件
- 用户指定本地仓库发生变化
- md文档不存在对应的英文翻译文档（`_en.md`），或存在文档变更

## 执行流程

### 1. 扫描所有md文档并统计

扫描用户本地仓库的所有`.md`文档（排除.git目录），**必须先统计排除路径文件数量，再计算实际需翻译数量**。

**扫描流程：**

1. **统计总md文件数量**
   ```bash
   find . -name "*.md" -not -path "./.git/*" -type f | wc -l
   ```

2. **统计排除路径文件数量**（整个docs目录）
   ```bash
   # 统计docs目录下的md文件数量
   find . -name "*.md" -not -path "./.git/*" -path "./docs/*" | wc -l
   ```

3. **计算实际需要翻译的文件数量**
   ```
   实际需翻译数量 = 总文件数 - docs目录文件数
   ```

**统计报告格式：**
向用户输出清晰的统计信息：
```
扫描结果：
- 总md文件：XX个
- 排除路径（跳过）：XX个
  - docs目录：XX个
- 实际需翻译：XX个（非docs目录下的md文件）
```

### 2. 判断文档是否需要翻译

对每个md文档按以下流程判断：

**判断文档是否需要翻译**
1. 非docs目录文档 → 需要翻译
2. docs目录文档 → 跳过（整个docs目录已排除）
3. 已存在翻译文件 → 检查是否有变更
   - 有变更 → 需要增量翻译
   - 无变更 → 跳过
4. 不存在翻译文件 → 需要完整翻译

> 详细判断流程和示例见 [Q1: 如何判断文档是否需要翻译](#q1-如何判断文档是否需要翻译)

**重要提示：**
- amct仓库的整个docs目录已排除，不需要翻译
- 仅翻译非docs目录下的md文档

### 3. 确定翻译文件存放路径

**非docs目录文档（仅翻译此类）：**
- 存放规则：与原文件同级目录
- 文件命名：`<原文件名>_en.md`
- 示例：`README.md` → `README_en.md`

**PR模板文件（特殊规则）：**
- 识别规则：文件名包含`PULL_REQUEST_TEMPLATE`且后缀为语言代码（如`.zh-CN.md`）
- 存放规则：与原文件同级目录（`.gitcode/`目录）
- 文件命名：将语言代码替换为对应英文代码，不使用`_en`后缀
- 示例：
  - `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md` → `.gitcode/PULL_REQUEST_TEMPLATE.en-US.md`
  - 其他平台模板：`ISSUE_TEMPLATE.zh-CN.md` → `ISSUE_TEMPLATE.en-US.md`

> 详细路径说明和示例见 [Q2: 如何确定翻译文件存放路径](#q2-如何确定翻译文件存放路径)

### 4. 检查是否存在翻译文件

检查非docs目录文档的翻译文件是否存在。

> 详细检查方法和示例见 [Q4: 如何检查是否存在翻译文件](#q4-如何检查是否存在翻译文件)

### 5. 检查文档变更

如果翻译文件已存在，检查原文档是否有变更。

> 详细检查方法见 [Q5: 如何检测文档变更](#q5-如何检测文档变更)

### 6. 读取翻译标准（必须执行）

> **⚠️ 重要警告：翻译前必须先读取翻译标准！遗漏此步骤会导致翻译不符合规范，需要返工修正！**

**必须在翻译任何内容之前**，读取`https://developers.google.com/style`英文风格指南的关键内容。

#### 翻译标准核心要点（必须遵守）

**一、语态规范**

| 规范 | 要求 | 示例 |
|------|------|------|
| 主动语态优先 | 面向用户的资料以主动语态为主 | ❌ "Designed for..." → ✅ "This guide provides..." |
| 操作类用祈使句 | 操作步骤省去you，直接使用动词开头 | ❌ "You can enter the password" → ✅ "Enter the password" |
| 被动语态例外 | 动作执行者未知/无关、错误提示中避免责备用户时可用被动 | "The dialog box is displayed" |

**二、时态规范**

| 场景 | 使用时态 | 示例 |
|------|---------|------|
| 陈述规律/原理/机制 | 一般现在时 | "A ping command sends packets to test connectivity." |
| 操作后的瞬时结果 | 一般现在时 | ❌ "The dialog box will appear" → ✅ "The dialog box appears" |
| 需间隔较长时间的结果 | 将来时 | "The system will restart after installation." |
| 已完成的动作 | 现在完成时 | "You have successfully logged in." |

**三、词汇规范**

| 禁止使用 | 正确用法 |
|---------|---------|
| etc. | and so on（需限定范围） |
| e.g. | for example |
| i.e. | that is |
| via | through / by / using |
| can't, it's, don't | cannot, it is, do not |
| you're, they're | you are, they are |
| won't, shouldn't | will not, should not |

**四、句子和段落规范**

| 规范 | 要求 |
|------|------|
| 重要信息置前 | 将关键信息放在句首或段落开头 |
| 避免超长句子 | 每句不超过25个单词 |
| 使用并行结构 | 相似描述使用统一句式 |
| 避免双重否定 | 使用直接陈述 |

**五、好的英文风格必须满足**

- 使用前后一致的术语
- 使用简单词汇
- 定义缩略语（首次出现时定义）
- 尽量使用主动语态
- 尽量使用一般现在时态
- 使用并行结构
- 使用第二人称（操作类用祈使句）
- 清晰、正确地组织信息

**六、好的英文风格必须避免**

- 虚悬前置词
- 无谓重复或累赘
- 外来语（etc、e.g.、i.e.、via）
- 过时词汇（thus、hereinafter、hence）
- 口语词汇（figure out）
- 词汇简缩（char、config）
- 缩略词（can't、it's）

**七、语法完整性检查（新增，重要）**

> ⚠️ 警告：中文省略主语常见，但英文必须有明确主语和动词！忽略此规则会导致严重语法错误！

**必须检查以下语法错误：**

- **缺少主语**
   - 中文省略主语常见，但英文必须有主语
   - 检查方法：每个句子是否包含明确的主语（you、the system、this guide等）
   

- **缺少动词**
   - 检查方法：每个句子是否包含明确的动词


- **句子结构完整性**
   - 使用"For + 动名词"结构引导长句
   - 避免"名词 + please refer to"的错误结构


**八、句子结构重构规则（新增）**

中文长句的英文处理策略：

- **"请参考..."句型**
   - ❌ 错误：直接翻译为"please refer to"放在句尾
   - ✓ 正确：使用"For..."引导或拆分为两个句子
   
   | 中文 | ❌ 错误 | ✓ 正确 |
   |-----|--------|--------|
   | XX操作请参考《指南》中的"准备软件包"章节 | "XX operations please refer to 'Prepare Software Package' chapter in Guide" | "For XX operations, refer to the 'Prepare Software Package' chapter in the Guide" |

- **条件句处理**
   - ❌ 错误：省略主语，直接用"If need..."
   - ✓ 正确：完整主语"If you need..."
   
   | 中文 | ❌ 错误 | ✓ 正确 |
   |-----|--------|--------|
   | 若仅编译算子，可以不安装 | "if only compiling operators, can not install" | "if you are only compiling operators, they do not need to be installed" |

- **并列句处理**
   - 使用分号连接相关句子
   - 或拆分为独立句子

**九、语态选择规则（补充）**

原则：主动语态优先，但需区分场景。

| 场景 | 推荐语态 | 示例 |
|------|---------|------|
| 用户操作指引 | 祈使句（主动） | "Select a software installation method..." ✓ vs "Software installation method selection..." ❌ |
| 功能描述 | 主动语态 | "WebIDE provides..." ✓ vs "WebIDE can provide..." ❌ |
| 状态描述 | 可用被动 | "The necessary software packages are already installed" ✓ |
| 条件说明 | 祈使句或you为主语 | "If you need to run samples..." ✓ vs "If need to run samples..." ❌ |

**对比示例：**

| 中文原文 | ❌ 错误翻译 | ✓ 正确翻译 | 分析 |
|---------|-----------|-----------|------|
| WebIDE可提供... | "WebIDE can provide..." | "WebIDE provides..." | 主动语态更直接 |
| 该平台为您提供... | "provides...for you" | "provides you with..." | "provide you with"更地道 |

**十、冠词使用规范（新增）**

必须检查冠词：

- **单数可数名词前必须有冠词**
   
   | 错误 | 正确 |
   |-----|------|
   | "WebIDE development platform" | "the WebIDE development platform" |
   | "Ascend environment" | "an Ascend environment" |
   | "Docker engine" | "the Docker engine" |

- **特指名词前用the**
   - "the host machine"（特指宿主机）
   - "the root user"（特指root用户）
   - "the CANN software package"（特指某个包）

- **泛指名词前用a/an**
   - "an Ascend environment"（泛指一个环境）
   - "a compilation environment"（泛指编译环境）

**十一、表格字段翻译标准（新增）**

**常用字段标准翻译：**

| 中文 | ✓ 标准翻译 | 说明 |
|------|-----------|------|
| 注意事项 | "Precautions" | 比"Note"更专业 |
| 说明 | "Description" | 标准用法 |
| 必选 | "Required" | 标准用法 |
| 可选 | "Optional" | 标准用法 |
| 建议 | "Recommended" | 比"Suggestion"更常见 |

**表格内容翻译要求：**
- 每个单元格必须是完整句子或短语
- 不可出现语法错误的片段

**十二、术语一致性表（新增）**

  **必须保持一致的术语：**

| 中文术语 | 标准英文翻译 | 备注 |
|---------|------------|------|
| 样例 | sample | 不使用example（不规范） |
| 环境 | environment | - |
| 部署 | deployment | - |
| 安装 | installation | install为动词，installation为名词 |
| 编译 | compilation/compile | compilation为名词，compile为动词 |
| 运行 | run/running | - |
| 宿主机 | host machine | 不使用host（不完整） |
| 容器 | container | - |
| 镜像 | image | - |
| 算子 | operator | - |
| 固件 | firmware | - |
| 驱动 | driver | - |

### 7. 执行翻译

**不存在翻译文件（完整翻译）：**
1. 使用read工具读取中文文档完整内容
2. 参考翻译标准进行完整翻译
3. 根据文档路径类型确定存放位置并创建翻译文件

**已存在翻译文件且有变更（增量翻译）：**
1. 使用bash工具获取中文文档的变更内容（diff）
2. 分析变更点
3. 将变更点翻译成英文
4. 使用edit工具更新对应的英文文档

> 详细处理方法见 [Q8: 如何处理已存在的翻译文件](#q8-如何处理已存在的翻译文件)

### 8. 使用todowrite跟踪进度

创建任务列表跟踪翻译进度，**必须包含统计步骤**（避免误导用户）。

**示例（正确统计流程）：**
```
[
  {"content": "扫描所有md文档并统计总数", "status": "in_progress", "priority": "high"},
  {"content": "统计排除路径文件数量（docs目录）", "status": "pending", "priority": "high"},
  {"content": "计算实际需翻译数量（总数-排除数）", "status": "pending", "priority": "high"},
  {"content": "读取翻译标准文档（必须执行）", "status": "pending", "priority": "high"},
  {"content": "翻译根目录核心文档（README/CONTRIBUTING/SECURITY等）", "status": "pending", "priority": "high"},
  {"content": "提交翻译变更到远程仓库", "status": "pending", "priority": "high"}
]
```

**统计报告示例：**
扫描完成后，向用户输出清晰的统计信息：
```
扫描结果：
- 总md文件：XX个
- 排除路径（跳过翻译）：XX个
  - docs目录：XX个
- 实际需翻译：XX个（非docs目录下的md文件）
- 已存在翻译文件：X个
- 待翻译文件：XX个
```

### 9. 提交变更到远程仓库

翻译完成后，将所有变更的中英文md文档提交到远程仓库。

**提交流程：**
1. 查看所有变更的md文档
2. 添加变更文件到暂存区
3. 创建提交记录
4. 推送到远程仓库

> 详细提交方法见 [Q9: 如何提交翻译变更到远程仓库](#q9-如何提交翻译变更到远程仓库)

## 配置文件说明

### 翻译标准文档
- **主文档**：https://developers.google.com/style

### 排除规则说明
- **amct仓库**：整个docs目录已排除，不需要翻译
- **仅翻译**：非docs目录下的md文档

## 目录结构示例

```
仓库根目录/
├── README.md                  # 原始中文文档（非docs）
├── README_en.md               # 翻译后英文文档（同级目录）
├── CONTRIBUTING.md            # 原始中文文档（非docs）
├── CONTRIBUTING_en.md         # 翻译后英文文档（同级目录）
├── docs/                      # docs目录（已排除，不翻译）
│   └── ...                    # docs下所有内容不翻译
├── examples/
│   └── README.md              # 原始中文文档（非docs）
│   └── README_en.md           # 翻译后英文文档（同级目录）
└── npu_ops/
    └── README.md              # 原始中文文档（非docs）
    └── README_en.md           # 翻译后英文文档（同级目录）
```

**说明：**
- docs目录下的所有md文档已排除，不需要翻译
- 仅翻译非docs目录下的md文档
- 翻译文件与原文件同级目录，命名为`<原文件名>_en.md`


## 工具使用建议

### 并行操作提高效率
- 并行读取多个相关文件（中文原文、英文版本、风格指南等）
- 并行执行多个bash命令获取不同信息
- 如果每次翻译的文件数量较多，则可以hi每20个文件分批处理；如果一个文件中行数超过300行，则可以每200行分批翻译

## 注意事项

 1. **扫描范围与统计（重要）**：
    - 扫描所有md文档，包括docs目录和非docs目录
    - **必须先统计排除路径文件数量**（docs目录），再计算实际需翻译数量
    - 向用户输出清晰的统计信息（总数、排除数、实际需翻译数）
    - 避免误导用户认为需要翻译所有文件
 2. **排除规则**：docs目录下的所有文档已排除，不需要翻译
 3. **必须先读取翻译标准**：翻译任何内容前，必须先读取英文风格指南
 4. **文件存放路径**：
    - 非docs目录：与原文件同级目录，命名为`<原文件名>_en.md`
    - docs目录：已排除，不翻译
 5. **文件命名规则**：
    - 非docs目录：`<原文件名>_en.md`
 6. **增量更新**：已存在翻译文件时，仅翻译变更部分，避免重复翻译
 7. **格式一致性**：保持Markdown格式、表格结构、链接引用等一致
 8. **提交变更**：翻译完成后必须将所有变更的中英文md文档提交到远程仓库
 9. **语法完整性检查（新增，重要）**：
    - 每个句子必须有明确主语（中文省略主语常见，英文必须有主语）
    - 每个句子必须有明确动词（避免"名词+please refer to"结构）
    - 条件句必须完整（使用"If you..."而非"If need..."）
    - 使用"For..."引导长句而非直接翻译"请参考"
 10. **避免冗余表达（新增）**：
    - 去掉冗余的"please"（技术文档应简洁直接）
    - 使用祈使句而非"You can..."结构
    - 表格字段使用专业术语（"Precautions"而非"Note"）

## 翻译质量检查清单（新增，必须执行）

> **⚠️ 重要：翻译完成后必须执行以下检查，确保翻译质量！**

**翻译完成后必须执行以下检查：**

1. **语法完整性检查**
   - [ ] 每个句子是否有明确主语（you、the system、this guide等）
   - [ ] 每个句子是否有明确动词
   - [ ] 条件句是否完整（If you...而非If need...）
   - [ ] 并列句是否正确连接（使用分号或拆分为独立句子）
   - [ ] "请参考"句型是否使用"For..."引导

2. **词汇规范检查**
   - [ ] 是否无违禁外来语（via, e.g., i.e., etc.）
   - [ ] 是否无缩写（can't, it's）
   - [ ] 是否无口语词汇（figure out）
   - [ ] 是否无冗余表达（去掉多余的please）

3. **语态检查**
   - [ ] 操作指引是否使用祈使句（Select...而非You can select...）
   - [ ] 功能描述是否使用主动语态（WebIDE provides...而非WebIDE can provide...）
   - [ ] 避免过度被动语态
   - [ ] 条件说明是否使用you为主语

4. **冠词检查**
   - [ ] 单数可数名词前是否有冠词（the/a/an）
   - [ ] 特指名词前是否用the（the host machine、the root user）
   - [ ] 泛指名词前是否用a/an（an Ascend environment）

5. **术语一致性检查**
   - [ ] 关键术语是否统一翻译（sample而非example）
   - [ ] 是否使用标准术语表（见"术语一致性表"）
   - [ ] 名词/动词形式是否正确（installation vs install）

6. **句子结构检查**
   - [ ] 长句是否合理拆分或使用"For..."引导
   - [ ] 是否无"名词+please refer to"错误结构
   - [ ] 是否无中文式句子结构（如"Host machine...operations please refer to..."）

7. **表格检查**
   - [ ] 表格字段是否使用标准翻译（Precautions而非Note）
   - [ ] 每个单元格是否为完整句子或短语
   - [ ] 是否无语法错误的片段

8. **格式检查**
   - [ ] Markdown格式是否正确
   - [ ] 表格结构是否完整
   - [ ] 链接是否有效
   - [ ] 代码块格式是否正确

**检查示例对比：**

| 问题类型 | ❌ 错误示例 | ✓ 正确示例 | 检查项 |
|---------|-----------|-----------|--------|
| 缺少主语 | "can directly use..." | "you can directly use..." | 语法完整性-主语 |
| 缺少动词 | "Software installation method selection..." | "Select a software installation method..." | 语法完整性-动词 |
| 条件句不完整 | "If need to run samples..." | "If you need to run samples..." | 语法完整性-条件句 |
| 句子结构混乱 | "Host machine...operations please refer to..." | "For downloading and installing..., refer to..." | 句子结构 |
| 冗余表达 | "please refer to..." | "refer to..." | 词汇规范-冗余 |
| 表格字段 | "Note" | "Precautions" | 表格检查 |
| 缺少冠词 | "WebIDE development platform" | "the WebIDE development platform" | 冠词检查 |
| 术语不一致 | "example" | "sample" | 术语一致性 |

## 常见问题处理

### Q1: 如何判断文档是否需要翻译

**判断流程：**

```
1. 文档路径是否以docs/开头？
   → 是：跳过（整个docs目录已排除）
   → 否：需要翻译（非docs目录文档）

2. 是否存在对应的翻译文件？
   → 否：需要完整翻译
   → 是：检查是否有变更

3. 文档是否有变更？
   → 是：需要增量翻译更新
   → 否：跳过
```

**检查命令：**
```bash
# 1. 统计总md文档数量（排除.git目录）
find . -name "*.md" -not -path "./.git/*" -type f | wc -l

# 2. 统计docs目录下的md文件数量（排除路径）
find . -name "*.md" -not -path "./.git/*" -path "./docs/*" | wc -l

# 3. 检查文档是否在docs目录
echo $filepath | grep "^./docs/"
```

**示例场景：**

| 文档路径 | 是否翻译 | 说明 |
|---------|---------|------|
| `README.md` | 翻译 | 非docs目录 |
| `examples/README.md` | 翻译 | 非docs目录 |
| `docs/api/guide.md` | 跳过 | docs目录已排除 |
| `docs/README.md` | 跳过 | docs目录已排除 |

### Q2: 如何确定翻译文件存放路径

**判断流程：**

```
文档路径是否在docs目录下？
→ 是：跳过（docs目录已排除）
→ 否：非docs目录文档，存放规则：与原文件同级目录，命名为<原文件名>_en.md
```

**检查命令：**
```bash
# 检查文档是否在docs目录下
echo $filepath | grep "^./docs/"
```

**存放规则总结：**
```bash
# 非docs目录：检查同级目录
ls <文件所在目录>/<文件名（不含扩展名）>_en.md
```

### Q3: 如何检查是否存在翻译文件

**判断流程：**

```
文档路径是否在docs目录下？
→ 是：跳过（docs目录已排除）
→ 否：检查同级目录下的<原文件名>_en.md文件
```

**检查命令示例：**

```bash
# 检查同级目录下是否存在_en.md文件
ls README_en.md
ls examples/README_en.md
ls npu_ops/guide_en.md
```

### Q4: 如何检测文档变更

**方式1：与上一次提交比较**
```bash
git diff HEAD~1 -- README.md
git diff HEAD~1 -- docs/api/quantize.md
```

**方式2：检查工作区状态**
```bash
git status README.md
git status docs/api/quantize.md
```

**方式3：获取变更统计**
```bash
git diff HEAD~1 --stat -- README.md
git diff HEAD~1 --stat -- docs/api/quantize.md
```

### Q5: 翻译时保留中文链接

对于引用其他中文文档的链接，保持原链接路径不变，待对应文档翻译后再更新。

### Q6: 为什么必须先读取翻译标准

**不读取翻译标准会导致以下问题：**

1. **被动语态滥用**：中文常见"面向..."、"设计用于..."等表达，直接翻译为"Designed for..."是被动语态，应改为主动语态"This guide provides..."

2. **外来语使用**：中文常用"等"、"例如"、"即"，直接翻译为etc、e.g.、i.e.是外来语，应改为and so on、for example、that is

3. **缩写形式**：翻译中使用can't、it's等缩写，应改为cannot、it is

4. **时态错误**：操作步骤后的结果使用将来时"will appear"，应改为一般现在时"appears"

**后果：** 翻译完成后发现不符合规范，需要返工修正，浪费时间。因此必须在翻译前先读取标准。

### Q7: 如何处理已存在的翻译文件

**处理流程：**
1. 检查原文档是否有变更
2. 如有变更，分析变更内容
3. 仅翻译变更部分
4. 使用edit工具更新同级目录的翻译文件

**增量翻译示例：**
```bash
# 1. 获取变更内容
git diff HEAD~1 -- README.md

# 2. 分析变更点
# 3. 翻译变更部分
# 4. 使用edit工具更新翻译文件
```

### Q8: 如何提交翻译变更到远程仓库

**提交流程：**

1. **查看所有变更的md文档**
```bash
# 查看所有变更文件
git status

# 查看变更的md文档（包括中文原文和英文翻译）
git status -- "*.md"
```

2. **添加变更文件到暂存区**
```bash
# 方式1：添加所有md文档变更
git add "*.md"

# 方式2：添加特定文件
git add README.md README_en.md
git add docs/api/quantize.md docs/en/api/quantize_en.md
```

3. **创建提交记录**
```bash
# 创建提交（推荐提交信息格式）
git commit -m "docs: add/update English translation for markdown files"

# 或更详细的提交信息
git commit -m "docs: translate README.md to README_en.md

- Add README_en.md (English translation)
- Update docs/api/quantize.md and docs/en/api/quantize_en.md"
```

4. **推送到远程仓库**
```bash
# 推送到远程仓库
git push origin <branch_name>

# 如果是新分支，使用 -u 参数
git push -u origin <branch_name>
```

**完整提交示例：**
```bash
# 1. 查看变更
git status

# 2. 添加所有md文件变更
git add "*.md"

# 3. 创建提交
git commit -m "docs: add English translation for markdown files"

# 4. 推送到远程
git push origin main
```

**注意事项：**
- 提交前确保所有翻译文件都已创建或更新
- 提交信息应清晰说明翻译的内容
- 中文原文和英文翻译应一起提交，保持同步
- 如有多个翻译文件，建议批量提交
