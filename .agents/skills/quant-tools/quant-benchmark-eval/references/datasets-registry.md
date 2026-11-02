# benchmark 注册表（`scripts/datasets.json`）

## 1. entry 格式

benchmark 任务注册给两类消费者：`prefetch_data.py` 按它下载，本地读取 shim（`bench_offline.py`）按它把 lm-eval 的数据请求翻译成本地文件；lm-eval 本身不读它。每个任务一条，键就是 `--task` 传的任务名：

```jsonc
"<task>": {
  "repo_id": ...,          // 从哪下（HF 数据集仓）
  "subdir": ...,           // 落到哪（$BENCH_ASSETS/<subdir>/）
  "dataset_name": ...,     // lm-eval 怎么命中（可省，见 §4）
  "splits": { "<split名>": <路径/glob 声明> },   // 命中后读什么（见 §3）
  "allow_patterns": [...]  // 下载白名单（可省，省了用默认 ["*.parquet","*.json","*.jsonl","*.md"]）；声明的 glob 要写在它覆盖范围内
}
```

## 2. 字段语义

| 字段 | 含义 |
|---|---|
| `repo_id` | HF 数据集仓，`prefetch_data.py` 拿它 `snapshot_download`，本地读取 shim 也拿它建键 |
| `subdir` | 缓存落地的子目录，最终路径是 `$BENCH_ASSETS/<subdir>/` |
| `dataset_name` | lm-eval 调 `datasets.load_dataset(path=..., name=...)` 的第二个值；**可省略**（见 §4） |
| `splits` | `split 名 → 相对 subdir 的路径 / glob`，值可为单字符串、字符串列表、`{"any_of": [...]}` 或 `{"all_of": [...]}` |
| `allow_patterns` | 下载白名单，**可省略**（省了用默认 `["*.parquet", "*.json", "*.jsonl", "*.md"]`，`mmlu_pro` 就没写）；声明的 glob 要写在它覆盖范围内 |

## 3. `any_of` 与 `all_of` 不能混

`splits` 的值两种写法，模式由数据集性质决定，不是口味问题。注册表里两条真实声明对照：

```jsonc
// math500 —— 同一份数据的候选布局：命中哪个读哪个，只读一份
"test": { "any_of": ["test.jsonl", "data/test.jsonl", "test-*.parquet",
                      "data/test-*.parquet", "**/test-*.parquet"] }

// livecodebench_release_v6 —— 累积分片：全部合并，缺一硬失败
"test": { "all_of": ["test.jsonl", "test2.jsonl", "test3.jsonl",
                      "test4.jsonl", "test5.jsonl", "test6.jsonl"] }
```

- `{"any_of": [...]}` —— 同一份数据的不同**候选布局**（MATH-500 既可能发 `test.jsonl`，也可能发 `data/test-*.parquet`）：按声明顺序取**第一个命中**的候选，其余忽略。缓存里同时存在两种布局（上游迁移 / 旧缓存残留）时也只读一份，样本不会翻倍。**累积集（LiveCodeBench 那类分片）一律用 `all_of`**：声明成 `any_of` 只读第一片，样本不全而分数照样出。
- `{"all_of": [...]}`（或裸字符串 / 裸列表）—— 必须**全部合并**的分片：缺任一片即硬失败，不拿子集凑数。

## 4. shim 的键是 `(repo_id, dataset_name)`，不是任务名

shim 按这对值建表（`bench_offline.py` 消费），取的是 lm-eval 传给 `datasets.load_dataset` 的那两个实参。entry 里的 `repo_id` / `dataset_name` **等于 lm-eval 实际会传的那对值**。

```jsonc
// arc_challenge —— dataset_name 写了：与 arc_easy 共用 allenai/ai2_arc 仓，靠它区分
"dataset_name": "ARC-Challenge"

// math500 —— 整个字段不写：lm-eval 虽传 default，(path, None) 回落也命中
//（注册表里没有 "dataset_name" 这一行）
```

- 多个任务共用一个仓是允许的（`arc_challenge` / `arc_easy` 共用 `allenai/ai2_arc`），前提是 `dataset_name` 把它们区分开。
- `dataset_name` 允许回落到 `(path, None)`：lm-eval 声明了 name 而注册表按仓维度不写 name 时也算命中。MATH-500 就靠它（task 声明 `dataset_name: default`，注册表不写），不需要为它多注册一条。
- 两个任务**共用同一对键却声明不同本地数据**：建表时抛 `ValueError`（点名两个任务与该键）。声明完全相同的重复项仍允许。
