# Paper / Plan Feature Audit — promised-but-not-done (AMCT pruning, 2026-06)

대조 소스: `PRUNING_STANDARD.md` §1/§6, QAP 딥리서치 메모(Bayesian Bits/DJPQ/GETA/Progressive-Intensity),
유지보수자 요구(prune↔quant harmony), 그리고 실제 코드(`amct_pytorch/experimental/pruning/`).
구현 여부는 코드 grep으로 확인함. (✅=있음, ⚠️=부분, ❌=없음)

## A. 표준(§6)이 "remaining"으로 명시했는데 미구현
| # | 기능 | 상태 | 근거 |
|---|---|---|---|
| A1 | **Size-budget 구동 프루닝** (목표 −X% 파라미터/MB로 비율 자동 탐색) | ❌ | 진입점은 `accuracy_based_auto_prune(tolerance=...)` 정확도-기반뿐. `JointBudget`은 prune/quant 사이 **정확도 tolerance 분배**이지 size 타깃 아님. 표준 §1은 "budget = target size OR accuracy"를 약속했으나 **size 모드 없음**. |
| A2 | **CNN 글로벌 size-budget + 레이어간 할당** | ⚠️ | `depgraph(global_pruning=True)` 존재하나 **size 예산 루프 없음**(고정 ratio). |

## B. QAP 리서치가 권고한 bridge인데 미구현 (논문 근거 있음)
| # | 기능 | 상태 | 논문/근거 |
|---|---|---|---|
| B1 | **Quant-aware saliency** — per-unit 중요도를 **양자화된 weight**로 점수 | ✅구현 / ⚠️**이득 없음(측정)** | `reconstruct(quant_aware=True, quant_aware_bits=N)` 추가(GETA식, fake-quant 후 점수). **그러나 실측**(SmolLM, prune0.3→전체 int4): blind ppl 367.97 vs aware 374.98 = **이득 없음**(int3는 둘 다 붕괴 10⁷~10⁹). 이유: per-channel 양자화가 컬럼 norm을 거의 안 바꿔 선택이 fp와 동일. → **opt-in 유지, 광고 안 함.** 실효를 보려면 saliency가 아니라 **양자화-인지 *재구성*** 또는 full-GETA joint-QAT 필요. |
| B2 | **Recalibrate-after-prune** (프루닝 후 양자화 scale 재유도) | ⚠️/❌ | 리서치 bridge (a) "always, cheap, safe". 코드에 **명시적 재캘리브 단계/옵션 없음**. amct.quantize가 fresh 캘리브를 하긴 하나, 파이프라인 가드/문서로 노출 안 됨. variance-saliency엔 페널티≈0(측정)이지만 **다른 saliency·per-tensor scale·aggressive ratio**에선 깨짐. |
| B3 | **Progressive-Intensity 순서 가드** (약교란 먼저: structured-prune→quant) | ⚠️ | 파이프라인은 그 순서지만 **명시적 권고/가드(API·문서)로 인코딩 안 됨**. (ICLR’26 PQQP) |

## C. 벤치마크가 실측으로 잡은 커버리지 갭
| # | 기능 | 상태 | 근거 |
|---|---|---|---|
| C1 | **Conv1D FFN 프루닝 지원** (GPT-2 계열) | ❌ | E2E 벤치: gpt2/distilgpt2/Cerebras-GPT prune **no-op** — FFN이 `transformers.Conv1D`라 dense 도메인 미탐지. pythia(GPTNeoX)/SmolLM(Llama, nn.Linear)은 정상. **dense 도메인에 Conv1D를 producer/consumer로 인식** 추가하면 GPT-2 전 계열 커버. |
| C2 | **Qwen2.5-0.5B int8 NPU 커널 실패** | ✅**해결(버그 아님)** | 격리 재현: ratio 0.2(keep 3891 비정렬)·0.25(keep 3648 정렬) **둘 다 quant OK**. 정렬 가설 **반증**. 원래 실패는 **동시-스윕 race(NPU 한 장에 2 프로세스 경합)** 아티팩트 → 단일 스윕으로 해소. Qwen 재추가·재실행. |

## D. 리서치에 "SOTA 프레이밍"으로 named 됐으나 본 PR 범위 밖(대형 미구현)
- **Bayesian Bits**(NeurIPS’20), **DJPQ**(ECCV’20), **GETA 전체**(CVPR’25), **Channel-wise MPS**(2024)
  = 미분가능 joint prune+quant(0-bit=pruned). 학습 기반 대공사 → 본 PR(무학습 PTQ-style 구조 프루닝) 범위 밖.
  *부분 차용*: B1(quant-aware saliency)이 GETA의 실용 핵심만 떼온 형태.

## 권고 우선순위 (작고 효과 큰 것부터)
1. **C1 Conv1D 지원** — 벤치가 실증한 실 갭, GPT-2 전 계열 즉시 커버. (작음·확실)
2. **B1 Quant-aware saliency** — 논문 근거, `quant_fn` 서사 완성(탐색뿐 아니라 선택까지 양자화-인지). (중간)
3. **A1 Size-budget 모드** — 표준이 약속한 핵심 누락(목표 −X%로 자동). (중간)
4. **B2 recalibrate 가드/옵션 노출 + 문서** — 안전망. (작음)
5. **C2 Qwen 커널 실패 추적** — 1모델 디버그. (작음)
