# ut-cov-report

单元测试覆盖率处理，支持两种模式：

- `cov`（默认）：按 `ut_process` 进一步区分
  - `ut_cov`：基于 `classify_rule.yaml` 统计增量覆盖率，打包 `ut_cov_<type>.tar.gz`。
  - `coverage`：查找 `ops.info`/`coverage.info` 并重命名为 `coverage_<type>.info` 供后续合并。
- `report`：从 OBS 拉取各模块覆盖率文件，合并（lcov）后生成增量覆盖率报告并打包。

## 输入参数

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `workspace` | 是 | - | 代码检出根目录 |
| `target_branch` | 是 | - | 目标分支 |
| `cov_mode` | 否 | `cov` | `cov`（打包覆盖率包）/ `report`（生成报告） |
| `coverage_num` | 否 | `80` | 覆盖率阈值 |
| `git_token` | 否 | - | 私有仓库访问 token |
| `ut_process` | 否 | - | 仅 cov 模式：`ut_cov` / `coverage`，空则跳过打包 |
| `ut_type` | 否 | - | cov 模式必填，UT 分类 |
| `ut_field` | 否 | - | report 模式必填，业务域标识 |
| `modules` | 否 | - | report 模式必填，空格分隔模块列表 |
| `source_branch` | 否 | - | report 模式必填，代码源分支 |
| `obs_path` | 否 | - | report 模式必填，OBS 文件上传路径 |

## 使用示例

```yaml
# 打包覆盖率（cov 模式，配合 UT 执行步骤的输出使用）
- name: ut_cov
  identifier: ut_cov
  uses: cann/.gitcode/actions/ut-cov-report@master
  with:
    ut_process: ${{ steps.ut.outputs.ut_process }}
    workspace: ${{ steps.process_checkout.outputs.path }}
    ut_type: ${{ inputs.ut_type }}
    target_branch: ${{ env.TARGET_BRANCH }}
    git_token: ${{ secrets.GIT_TOKEN }}

# 生成报告（report 模式，合并各模块覆盖率）
- name: ut_report
  identifier: ut-report
  uses: cann/.gitcode/actions/ut-cov-report@master
  with:
    workspace: ${{ steps.process_checkout.outputs.path }}
    cov_mode: "report"
    ut_field: "rts_common"
    modules: "rts_v201 rts_david rts_910b rts_common"
    source_branch: ${{ env.SOURCE_BRANCH }}
    target_branch: ${{ env.TARGET_BRANCH }}
    obs_path: ${{ env.obs_path }}
    git_token: ${{ secrets.GIT_TOKEN }}
```
