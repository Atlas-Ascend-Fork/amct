# AMCT 仓 Agent skills 规划

## AMCT 仓 skills 清单

> **说明**：清单中已勾选表示该 skill 已就绪，未勾选表示该 skill 尚在规划中、还未实现。

- [x] **gitcode-issue** — 读取 Issue 详情、读取和回复评论，触发指令 `读取issue 168，并提交pr修复`
- [x] **gitcode-pr** — 创建 PR、提交行内检视意见、cherry-pick 代码到商用分支，触发指令`检视pr 1437` 或 `创建pr到develop分支`
- [ ] **superpowers** — 需求开发（生成软件设计文档、编码、生成测试用例），触发指令示例：`开发需求，要求<用户补充需求描述>`
- [ ] **amct-reviewer** — 遵循各种编码规范、编码军规、模块软件设计约束检视代码
- [ ] **gitcode-pipeline** — 触发流水线任务、查询流水线状态、获取失败任务日志
- [ ] **amct-builder** — 编译和执行 UT/ST 用例
- [ ] **amct-tester** — 生成用例，在带有npu的环境上执行用例
- [ ] **install-cann-toolkit** — 下载最新cann toolkit包，安装

## agent要支持的流程

- 需求开发：完成从软件设计到编码再到验证完整流程，使用gitcode-pr提交pr，个人检视代码，gitcode-pipeline触发流水线，并定时获取结果，如果流水线失败，可获取对应失败任务日志，本地修改代码，再次提交pr监控流水线。
- 问题单修改：修改代码，后续提交pr流程与上面一致。
- 解决issue：使用gitcode-issue读取issue及评论，如果涉及修改代码或文档，与上述流程一致。
- 执行测试用例或样例：使用amct-tester在真实环境中执行用例

## AMCT 仓 skills 路径

- 项目组共享，希望能做到启动agent默认安装或更新
- 仅在amct仓使用的skills可直接提交到amct仓`.claude/skills`目录
- 多个仓共用的skills，源码在公共仓（当前是https://gitcode.com/cann-agent/skills），启动agent时会自动下载或者更新skills到`.claude/skills`目录

> **注意**：`gitcode-issue`、`gitcode-pr` 等跨仓共享的 skills 不在 AMCT 仓的 `.claude/skills` 目录中维护。它们的源码存放在公共仓，在 amct 目录下启动 agent 后会自动下载并安装到本地 `.claude/skills` 目录。

## Agent辅助流程
```mermaid
flowchart TB
    subgraph 入口节点
        A1[需求开发]
        A2[问题单修改]
        A3[Issue处理]
    end

    subgraph 判断
        B1{是否需要<br/>代码修改?}
        B2{是否需要<br/>设计阶段?}
    end

    subgraph 设计阶段
        C1[软件设计<br/>需求规格]
    end

    subgraph 编码验证阶段
        D1[编写代码]
        D2[执行ut/st]
        D3[amct-reviewer<br/>agent检视代码]
        D4[amct-tester<br/>本地验证]
    end

    subgraph PR流程
        E1[gitcode-pr<br/>创建PR]
        E2[个人代码检视]
        E3[gitcode-pipeline<br/>触发流水线]
        E4[gitcode-pipeline<br/>定时获取<br/>流水线结果]
    end

    subgraph 流水线结果处理
        F1{gitcode-pipeline<br/>流水线<br/>是否通过?}
        F2[gitcode-pipeline<br/>获取失败<br/>任务日志]
        F3[Agent本地修复]
    end

   subgraph 推送代码
        I1[gitcode-pr<br/>提交代码]
    end

    subgraph 完成
        H1[流程结束]
    end

    %% 入口到判断
    A1 --> B2
    A2 --> B1
    A3 --> B1

    %% 设计判断
    B2 -->|是| C1
    B2 -->|否| D1
    C1 --> D1

    %% 编码判断
    B1 -->|是| D1
    B1 -->|否| H1

    %% 编码到PR
    D1 --> D2
    D2 --> D3
    D3 --> D4
    D4 --> E1

    %% PR流程
    E1 --> E2
    E2 --> E3
    E3 --> E4
    E4 --> F1

    %% 流水线结果处理
    F1 -->|通过| H1
    F1 -->|失败| F2
    F2 --> F3
    F3 --> I1
    I1 --> E3
```
