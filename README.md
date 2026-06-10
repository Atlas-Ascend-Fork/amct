# community-tasks 分支

本分支(`community-tasks`)用于**社区任务代码提交**,集中管理与归档社区贡献者完成各项任务后产出的代码、实验脚本与结果。

## 目录结构

所有任务代码统一放在 `experiment/task-book/` 下,每个任务对应一个独立目录:

```
experiment/
└── task-book/                      # 社区任务集
    └── <task-name>_<gitcode_id>/   # 单个任务目录
        ├── *.py                    # 任务代码与实验脚本
        ├── README.md               # 任务说明
        ├── run.sh                  # 运行脚本(可选)
        └── result_*.json           # 实验结果(可选)
```

## 命名规范

任务目录命名为 `<task-name>_<gitcode_id>`:

- `task-name`:任务名称,清晰反映任务内容。
- `gitcode_id`:提交者的 GitCode 账号,可使用中文。

示例:`amct_experience_张三`

## 提交说明

- 每个任务单独建目录,不与其他任务混放。
- 目录内附 `README.md`,说明任务目标、运行方式与结果。
- 提交前确认代码可运行;运行脚本与结果文件可按需附上。
