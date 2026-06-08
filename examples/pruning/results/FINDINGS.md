# Model-Zoo E2E Pipeline — Findings (running)

Device: NPU 2 (ASCEND_RT_VISIBLE_DEVICES=2), real AMCT int8 quant. Stages: baseline → prune → prune+int8.
Metric: CNN=top-1 acc (MNIST), LM/MoE/HF=perplexity (WikiText held-out). Full table: `model_zoo_pipeline.md`.

## Key observations (first 22 models)

1. **int8 quantization is nearly free across the board.** prune→int8 barely moves the metric
   (e.g. gpt2 60.92→60.74 ppl; cnn_w32_d4 0.962→0.962 acc) while MB halves. Confirms the earlier
   prune×quant probe: int8 adds a tiny, near-constant penalty → quant-aware pruning (`quant_fn`)
   has marginal value at int8 (matters for int4 / tight tolerance).

2. **COVERAGE GAP — GPT-2 family FFN is not pruned.** distilgpt2 / gpt2 show prune = no-op
   (88.18→88.18, 60.92→60.92) because their FFN uses `transformers.Conv1D`, not `nn.Linear`,
   so the dense domain finds no targets and skips. GPTNeoX (pythia) and Llama (SmolLM) use
   `nn.Linear` FFN → prune fires. → **TODO: add Conv1D support to the dense domain** to cover GPT-2.

3. **MoE expert pruning (0.5) is costly on tiny under-trained MoEs.** ppl jumps (974→1479 .. 974→1748);
   removing half the experts removes real capacity. Expected; reconstruct does not apply to expert removal.

4. **CNN: bigger nets prune+quant near-losslessly**, small ones lose more
   (w32_d4 0.983→0.962→0.962 at 76% size↓; w16_d3 0.867→0.640). int8 ≈ prune (only the Linear head is
   quantized; AMCT does not quantize Conv2d).

5. **From-scratch tiny LMs have high ppl (~900–1180)** — they are small + short-trained (real but weak),
   so FFN pruning barely moves ppl (near-random FFN reconstructs trivially). For sharper LM deltas, train
   longer; current value is in the *trend/coverage*, not absolute ppl.

## Honest caveats
- `sshleifer/tiny-gpt2` is a randomly-initialized test model (ppl ~50k) — a degenerate data point, kept for coverage.
- Param count after quant is held at the pruned count (quant changes precision/bytes, not weight count);
  size shown as effective MB (int8=1B, fp16=2B per weight).

## CNN: torch_pruning 제거 + 자체 저파괴 알고리즘 (2026-06-09)

- vendored torch_pruning + cnn_depgraph 삭제 (codecheck 706 + SCA 위험 제거). cnn은 **자체 코드만**.
- 새 `cnn.reconstruct`(무 third-party): cnn 도메인의 (conv→bn→consumer) 순차 타겟 + im2col 최소제곱 재구성.
- **정직한 검증 (학습된 MNIST CNN, TinyCNN w24d3, prune 0.3, baseline acc 0.926):**
  - `variance_channel` (naive 切片): acc **0.343**
  - `cnn.reconstruct`: acc **0.574** (같은 params, **+23%p**) — 진짜 저파괴.
- 랜덤 모델에선 우월하지 않음(랜덤 가중치는 재구성 일반화 X) → 단위테스트는 구조정확성만 단언, 정확도 우월은 trained로 실측.

## 사일런트 바이패스 감사 (제거/표면화)
- **제거**: "재구성이 naive보다 나으면만 적용"하던 가드(조용한 no-op = 기만) 삭제.
- **표면화**: ① `cnn.reconstruct` 欠定/Linear consumer로 재구성 못하면 **경고+report**(naive로 위장 안 함)
  ② `accuracy_based_auto_prune`/`size_budget_prune`의 trial `except`가 **에러를 경고**(질 거부로 위장 안 함).
