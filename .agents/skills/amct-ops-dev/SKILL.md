---
name: amct-ops-dev
description: |
  面向 amct_ops 目录下昇腾 NPU 自定义算子（Ascend C kernel + PyTorch C++ extension + Python 包装）开发、构建、注册和调试的技能。用户要求新增/修改 NPU 算子、写 Ascend C kernel、写 torch 绑定、注册 torch.ops.amct.* schema、调 ops_build.sh、改 CMakeLists.txt、处理 NPU_ARCH/SOC 平台差异（A2/A3 dav-2201，A5 dav-3510）、UB 内存/tile 计算、wheel 打包，或排查 amct_ops 编译/运行时问题（含 torch.compile / 图模式下算子报 "no Meta kernel"、需要补 Meta 后端做形状推导）时使用。
---

# AMCT NPU 算子开发

本 skill 用于在 `amct_ops/` 中新增或修改 NPU 自定义算子。`amct_ops` 与 `amct_pytorch` 是清晰分层关系：`amct_pytorch` 处理算法与流程编排，`amct_ops` 只承载硬件实现，通过 `torch.ops.amct.<name>` 暴露给上层。开发时不要打破这个边界 —— 算法逻辑不应出现在 kernel/extension 里，kernel 也不应反过来 import `amct_pytorch`。

## 先读上下文

按改动范围读最小必要文件：

```bash
# 整体结构与构建入口
sed -n '1,160p' amct_ops/README.md
sed -n '1,200p' amct_ops/ops_build.sh
sed -n '1,80p'  amct_ops/setup.py
sed -n '1,30p'  amct_ops/ops_init.py

# 参考模板（hifloat8_cast）
sed -n '1,200p' amct_ops/hifloat8_cast/CMakeLists.txt
sed -n '1,120p' amct_ops/hifloat8_cast/op_kernel/hifloat8_cast_kernel.cpp
sed -n '1,80p'  amct_ops/hifloat8_cast/op_kernel/hifloat8_cast_tiling.h
sed -n '1,40p'  amct_ops/hifloat8_cast/op_extension/ops.h
sed -n '1,120p' amct_ops/hifloat8_cast/op_extension/register.cpp
sed -n '1,80p'  amct_ops/hifloat8_cast/op_extension/hifloat8_cast_torch.cpp
sed -n '1,40p'  amct_ops/hifloat8_cast/python/hifloat8_cast/__init__.py
sed -n '1,80p'  amct_ops/hifloat8_cast/python/hifloat8_cast/ops.py
```

## 目录约定与三层职责

每个算子目录 `amct_ops/<op>/` 都遵循同一布局，新增算子时**严格按这个布局复制 `hifloat8_cast/` 做模板**，不要自己发明新结构。打包脚本依赖此约定来汇集 `.so` 和 Python 包。

```
amct_ops/<op>/
├── CMakeLists.txt              # 编译入口；按 NPU_ARCH 切 ISA
├── op_kernel/
│   ├── <op>_kernel.cpp         # Ascend C kernel（NPU 侧执行）
│   └── <op>_tiling.h           # 主机/设备共用的 tiling 常量、enum
├── op_extension/
│   ├── ops.h                   # C++ host 接口声明（namespace AscendKernel）
│   ├── register.cpp            # TORCH_LIBRARY_FRAGMENT(amct, ...) 注册 schema + PrivateUse1 + Meta dispatch
│   └── <op>_torch.cpp          # Torch tensor ↔ ACL stream，调用 kernel host stub
└── python/
    └── <op>/                   # 安装后即 amct_ops.<op>
        ├── __init__.py         # 加载 .so + re-export
        └── ops.py              # 薄 Python 包装（dtype 检查、docstring）
```

**三层职责分工**：

| 层 | 关注点 | 不要做的 |
| --- | --- | --- |
| `op_kernel/` | UB 管理、tile 计算、向量/标量指令、双缓冲 | 调用 ACL、依赖 Torch 类型 |
| `op_extension/` | Torch tensor 解包、stream 获取、kernel host stub launch、schema 注册 | 写算法逻辑、改 tile 形状 |
| `python/` | dtype/shape 输入检查、文档、re-export `torch.ops.amct.<fn>` | 算子语义实现、绕过 schema |

## torch.ops 命名空间

所有 amct_ops 算子统一注册到 `amct` namespace，schema 在 `op_extension/register.cpp` 中用 `TORCH_LIBRARY_FRAGMENT(amct, m)` 定义。Python 侧两种调用等价：

```python
torch.ops.amct.<fn>(...)                 # 直接走 dispatcher
from amct_ops.<op> import <fn>           # 薄包装，多一次类型检查
```

注册要点（以 `register.cpp` 为模板）：

1. `TORCH_LIBRARY_FRAGMENT(amct, m)` 内 `m.def("<fn>(Tensor input, ...) -> Tensor")` 写 schema。多个算子可以分散在不同 op 的 `register.cpp`，**namespace 必须都是 `amct`**。
2. PrivateUse1（NPU）实现里用 `TORCH_CHECK(...)` 校验 dtype/shape，错误信息要把实际值打出来。
3. 用 `TORCH_LIBRARY_IMPL(amct, PrivateUse1, m)` 绑定实现到 NPU 后端。
4. **同时注册 Meta 后端**：`TORCH_LIBRARY_IMPL(amct, Meta, m)` 绑一个 shape-only 实现，只按输入推出输出的 shape/dtype 而不真正算（`return at::empty(input.sizes(), input.options().dtype(...))`）。这是让算子能走 `torch.compile`/fake tensor/`meta` 设备做形状推导的关键——没有它，图模式 tracing 到这个算子会报 "no Meta kernel"。模板 `register.cpp` 的 `EncodeMeta`/`DecodeMeta` 就是范例。
5. 不要再为 CPU/CUDA 注册 fallback —— 这些算子就是 NPU only，让没在 NPU 上的实算调用直接 dispatcher 报错（Meta 后端不算实算，它只推形状）。

## Kernel 层要点（Ascend C）

参考 `hifloat8_cast/op_kernel/hifloat8_cast_kernel.cpp` 的写法：

- **tile 形状由 host 计算**：`tiling.h` 定义结构体（如 `castMode`、`tileLength`、`tailLength`），host 侧根据 `GetCoreMemSize(...)` 算实际 UB 可用量，kernel 侧只读取传入的 tiling 参数。这样 A2 (UB 256KB) / A3 (UB 512KB) 同一份编译产物在运行时自适应。
- **UB 内存预算**：把每个 buffer 的字节数写在文件头注释里（`LUT + inQ + outQ + ...`），方便后续 reviewer 验算。`hifloat8_cast_kernel.cpp` 文件头的「UB 占用」段是规范模板。
- **数据搬运用双缓冲**：`DataCopy` 配合两个 inQ / outQ 实现 compute-copy overlap。
- **不要写死 blockDim/tileLength**：通过 tiling struct 传入，host 侧 launch 时填。

## Extension 层要点（PyTorch C++ binding）

参考 `op_extension/hifloat8_cast_torch.cpp` 的模式：

```cpp
auto stream = c10_npu::getCurrentNPUStream();   // 当前 NPU stream
// 计算 tileLength（按平台 UB 大小），构造 tiling struct
// 申请输出 tensor（at::empty_like 或 at::empty 指定 dtype）
// 调用 extern "C" 生成的 kernel host stub：
hifloat8_cast_kernel_lut(blockDim, /*l2Ctrl=*/nullptr, stream.stream(),
                         input.data_ptr<uint8_t>(), ...);
return output;
```

- `extern "C" void <op>_kernel_<variant>(...)` 是 ASC 编译器从 `op_kernel/` 自动生成的 host stub；CMakeLists 里 `project(<op> LANGUAGES ASC CXX)` 触发它。
- Stream 一律从 `c10_npu::getCurrentNPUStream()` 拿，不要自己 create stream。
- 形状/dtype 校验放在 `register.cpp` 的 `*Impl` 里（dispatcher 入口），不要塞到 kernel 内部。

## Python 层要点

参考 `python/hifloat8_cast/ops.py`：

```python
def encode_to_hifloat8(x: torch.Tensor) -> torch.Tensor:
    """<一句话功能> + dtype/形状契约 + Example。"""
    return torch.ops.amct.encode_to_hifloat8(x)
```

- 函数体**几乎只有一行 `torch.ops.amct.<fn>(...)` 透传**；额外的输入检查只在 schema 层不够时再加。
- Docstring 必须包含 dtype 约束、shape 约束、device 约束（NPU only）、Example。
- `__init__.py` 负责 `torch.ops.load_library(<so_path>)` 并 re-export 函数；模块装好后 `import amct_ops.<op>` 必须成功且自动注册到 `torch.ops.amct`。

## 构建与平台

构建一律走 `amct_ops/ops_build.sh`，不要直接调 `python setup.py` 或 `cmake`。脚本会处理：算子 CMake → `.so` → `staging/` → wheel。

```bash
cd amct_ops/
bash ops_build.sh                                  # 全部算子，默认 A2
bash ops_build.sh --soc ascend910_93               # A3
bash ops_build.sh --soc ascend950 <op>             # 指定算子 + A5
```

`--soc` 到 `NPU_ARCH` 的映射在 `ops_build.sh` 顶部注释和各算子 `CMakeLists.txt` 头部都有，**改任一方都要同步**：

| `--soc` | `NPU_ARCH` | 覆盖型号 | UB |
| --- | --- | --- | --- |
| `ascend910b`（默认） | `dav-2201` | A2: 910B1/B2/B3 | 256 KB |
| `ascend910_93` | `dav-2201` | A3: 910_93 | 512 KB |
| `ascend950` | `dav-3510` | A5 | 512 KB+ |

A2/A3 编译产物相同；UB 差异由运行时 `GetCoreMemSize()` 决定 `tileLength`。**新增算子的 kernel 必须做这件事**，否则 A3 上跑会浪费一半 UB。

### 已知告警

`find_package(Torch)` 可能输出 `kineto_LIBRARY-NOTFOUND` —— pip 安装的 PyTorch 缺 Kineto 静态库，amct_ops 不用 profiler，忽略即可。

## 新增算子流程

1. **复制模板**：`cp -r amct_ops/hifloat8_cast amct_ops/<new_op>`，删掉里面的 `build/`、`*.so`。
2. **改名**：在新目录里把所有 `hifloat8_cast` 替换成 `<new_op>`，包括 `CMakeLists.txt` 的 `project(...)`、文件名、`python/hifloat8_cast/` 目录名、Python 包内的导入。
3. **写 kernel**：在 `op_kernel/<new_op>_kernel.cpp` 实现算子逻辑，`*_tiling.h` 定义 host/device 共用结构。**先把 UB 预算写在文件头注释里**。
4. **写 binding**：
   - `ops.h` 声明 host 接口
   - `register.cpp` 写 schema + dtype check + dispatch
   - `<new_op>_torch.cpp` 解包 tensor、算 tile、launch kernel
5. **写 Python wrapper**：`python/<new_op>/__init__.py` 加载 `.so` 并 re-export；`ops.py` 写薄包装 + docstring。
6. **更新顶层 README**：在 `amct_ops/README.md` 的「支持的算子」表里加一行。
7. **构建验证**：`bash ops_build.sh <new_op>`，期望产出 `dist/amct_ops-*.whl` 且 `pip install` 后 `import amct_ops.<new_op>` 成功。
8. **加单测验证**：在 `tests/amct_ops/` 下加 `test_<op>.py`，在 NPU 机器上跑（详见下方「验证」小节），用最小输入对照 CPU 参考实现校验数值。

## 实现注意事项

- **不要给单个算子改 `ops_build.sh` / `setup.py` 的通用逻辑** —— 这两个是所有算子共用的；要做特殊处理放算子自己的 `CMakeLists.txt`。
- **`staging/` / `build/` / `dist/` 都是构建产物**，不要提交，不要在 PR diff 里出现。
- **kernel 内不要 `#include` 任何 Torch / ACL 头**；只用 Ascend C 头和 `tiling.h`。这是为了让 kernel 可以独立用 ASC 编译器跑端到端测试，不依赖 Torch 环境。
- **schema 改动是破坏性变更** —— `register.cpp` 里的 `m.def("...")` 一旦发版被外部依赖，再改签名（增减参数、改 dtype）会让下游崩。新行为优先用新函数名 + 旧函数保留。
- **dtype 校验三处一致**：Python `ops.py` docstring、`register.cpp` 里的 `TORCH_CHECK`、kernel `tiling.h` 里的 `castMode` 枚举 —— 三个地方都改，否则用户看到的报错和实际行为对不上。
- **避免在 Python 层做 reshape/contiguous**：让 schema 接收什么形状就直接给 kernel，kernel 内部按 element-wise 处理；需要非 element-wise 的语义在 schema 层就写清楚约束。

## 验证

amct_ops 的单测在独立目录 `tests/amct_ops/`（用 `unittest` 写，如 `test_hifloat8_cast.py`，运行方式见 `tests/amct_ops/README.md`）。验证按以下顺序做：

1. **构建产物 sanity**：`bash ops_build.sh <op>` 成功 + `unzip -l dist/amct_ops-*.whl | grep <op>` 能看到 `.so` 和 Python 文件。
2. **导入测试**：在 NPU 机器上 `python -c "import amct_ops.<op>; print(torch.ops.amct.<fn>)"`，确认 schema 注册成功。
3. **跑单测**：在 `tests/amct_ops/` 下给新算子加一个 `test_<op>.py`（参考 `test_hifloat8_cast.py`：`unittest.TestCase` + `setUpClass` 里 `torch.npu.set_device(0)`，造小张量调算子、和 CPU 参考实现对比）。运行用 staging（开发推荐）或装 wheel 后：

   ```bash
   bash amct_ops/ops_build.sh <op>
   PYTHONPATH=amct_ops/staging python3 -m unittest tests.amct_ops.test_<op>
   ```

   需先 source CANN 环境、当前机器可访问 NPU（测试会 `torch.npu.set_device(0)`）。

   > **延伸：脱离 torch 独立验 kernel。** 想不依赖 torch 环境、直接用 `<<<>>>` 直调跑 kernel 验数值，可参考官方 Kernel 直调模板 [cann/cannbot-skills `ops/ascendc-direct-invoke-template`](https://gitcode.com/cann/cannbot-skills/blob/master/ops/ascendc-direct-invoke-template/SKILL.md)。它是独立直调工程，与本 skill 的 amct_ops 集成算子（`torch.ops.amct.*` + wheel）形态不同，但 kernel 写法、TPipe/TQue 管 UB、CopyIn→Compute→CopyOut 三段式是通用的。本 skill 的 Ascend C 分层心智模型也借鉴自该模板。
4. **平台覆盖**：A2 上 `bash ops_build.sh --soc ascend910b <op>`，A3 上 `--soc ascend910_93`，A5 上 `--soc ascend950`。即使只有 A2 机器，至少保证 A3/A5 编译通过（产物文件存在）。

回复时说明：跑过哪些命令、构建/导入/数值是否通过、未覆盖的硬件平台。Kernel 数值正确性最终要在真实 NPU 上验证，不要假装本地编译通过就等于正确。
