# Base vs QLoRA · Gold Only 평가 결과 (qlora_gold_v001)

> 2026-09-29 · 브랜치 `lkh_train` · 커밋 `53a1221`
> Gemma4 E4B-it를 Gold 917건으로 QLoRA 튜닝하고, 튜닝 전 Base 모델과 같은 Gold Test 115건에서 비교했다.
> 원자료: `training/experiments/base_v001/`, `training/experiments/qlora_gold_v001/` (metrics.json · config.yaml · loss_history.csv)

---

## 요약

- QLoRA는 Base보다 **Aspect F1 +6.8%p (0.858 → 0.926)**, **Evidence 정확 일치 +27.7%p (0.510 → 0.787)**, **리뷰 완전 일치 +34.8%p (0.226 → 0.574)** 높다.
- 개선은 대부분 **정밀도**에서 나왔다. 모델이 없는 aspect를 지어내는 일이 줄었고, 재현율은 그대로다 (0.921 → 0.915).
- paired bootstrap 95% 신뢰구간에서 Aspect F1 · 정밀도 · Evidence 정확 일치 · 리뷰 완전 일치의 차이는 0을 포함하지 않는다.
- 남은 약점: 재현율, 그리고 사실만 말한 구절(정답 `neutral`)을 긍정·부정으로 판단하는 경향.

## 1. 실험 조건

| 항목 | Base (`base_v001`) | QLoRA (`qlora_gold_v001`) |
| --- | --- | --- |
| 모델 | `google/gemma-4-E4B-it` | 같은 모델 + LoRA 어댑터 |
| 가중치 정밀도 | bf16 (16bit) | 4bit NF4 (double quant, compute bf16) |
| 학습 데이터 | 없음 | Gold Train 917건 (`gold_split_v1`) |
| 평가 데이터 | Gold Test 115건 · aspect 164개 | 같음 |
| prompt · 디코딩 | `training/prompts/`, greedy, `max_new_tokens` 768, `enable_thinking: false` | 같음 |
| GPU | RTX 3060 12GB | 같음 |

QLoRA 하이퍼파라미터:

| 구분 | 값 |
| --- | --- |
| LoRA | r 16 · alpha 32 · dropout 0.05 · 텍스트 타워 q/k/v/o/gate/up/down_proj (258개 모듈) |
| 학습 | lr 2e-4 · 3 epoch · batch 2 × 누적 8 · cosine · warmup 3% · weight decay 0 · paged_adamw_8bit · seed 42 |

- 분할: `place_id` 단위 group split이다. Train · Validation · Test에 같은 장소가 없다 (`training/data/README.md`).
- 지표 정의: `docs/evaluation-plan.md` 2절과 `src/travel_planner/finetune/evaluation/metrics.py` 맨 위 주석을 따른다.

## 2. 결과

### 2-1. 전체

| 지표 | Base | QLoRA | 변화 |
| --- | ---: | ---: | ---: |
| Aspect F1 (micro) | 0.858 | **0.926** | +6.8%p |
| Aspect 정밀도 | 0.803 | **0.938** | +13.5%p |
| Aspect 재현율 | 0.921 | 0.915 | -0.6%p |
| Aspect macro F1 | 0.740 | 0.784 | +4.4%p |
| Attribute 정확도 | 0.974 | 0.993 | +1.9%p |
| Sentiment 정확도 | 0.954 | 0.967 | +1.3%p |
| Sentiment macro F1 | 0.755 | 0.813 | +5.8%p |
| Evidence 원문 포함률 | 0.979 | **1.000** | +2.1%p |
| Evidence 정확 일치 | 0.510 | **0.787** | +27.7%p |
| Evidence 겹침 F1 | 0.870 | 0.925 | +5.5%p |
| JSON 파싱 성공률 | 1.000 | 1.000 | 0 |
| JSON 준수율 (허용값 포함) | 0.965 | 0.991 | +2.6%p |
| 리뷰 완전 일치 | 0.226 | **0.574** | +34.8%p |
| traveler_context F1 | 0.286 | 0.667 | 표본 2건 (4절) |
| 빈 리뷰 정확도 | 0.250 | 1.000 | 표본 4건 (4절) |

예측 aspect 수는 188개에서 160개로 줄었지만, 짝지어진(맞힌) aspect는 151개에서 150개로 거의 같다. 줄어든 28개는 대부분 없는 aspect를 지어낸 예측(FP)이었다.

### 2-2. 카테고리별 Aspect F1

| 카테고리 | Test 리뷰 / aspect | Base | QLoRA | 변화 |
| --- | --- | ---: | ---: | ---: |
| 호텔 | 38 / 60 | 0.915 | 0.951 | +3.6%p |
| 식당 | 39 / 61 | 0.853 | 0.917 | +6.4%p |
| 관광지 | 38 / 43 | 0.787 | 0.902 | **+11.5%p** |

관광지 개선이 가장 크다. 관광지의 정밀도가 0.725에서 0.949로 올랐다. 재현율은 세 카테고리 모두 Base와 같거나 거의 같다.

### 2-3. 통계적 유의성 (paired bootstrap)

같은 Test 리뷰를 리뷰 단위로 복원추출해 2,000번 다시 뽑았다 (seed 0). 그리고 매번 두 모델의 지표 차이(QLoRA − Base)를 계산했다.

| 지표 | 차이의 95% 신뢰구간 | 판정 |
| --- | --- | --- |
| Aspect F1 | [+0.019, +0.122] | 유의미 |
| Aspect 정밀도 | [+0.083, +0.191] | 유의미 |
| Evidence 정확 일치 | [+0.193, +0.360] | 유의미 |
| 리뷰 완전 일치 | [+0.252, +0.443] | 유의미 |
| Sentiment 정확도 | [-0.001, +0.035] | 판단 불가 |
| Attribute 정확도 | [+0.000, +0.045] | 판단 불가 |

Sentiment · Attribute는 Base도 이미 95% 이상이라 이 표본 크기로는 차이를 가릴 수 없다. 이 bootstrap 계산은 아직 평가 코드에 들어 있지 않다. `training/artifacts/<run>/predictions/test.jsonl`에서 일회성 스크립트로 계산했다.

## 3. 오류 분석 (QLoRA)

### 3-1. 놓친 aspect (FN 14개)

cleanliness 2, food_quality 2, 그리고 scenery · activity_variety · amenities · freshness · parking_availability · serving_speed · family_friendly · photo_spots · walking_burden · parking_experience가 1개씩이다. 특정 aspect에 몰리지 않고 흩어져 있다.

### 3-2. 지어낸 aspect (FP 10개)

seating_comfort 2, 그리고 bathroom_quality · walking_burden · amenities · parking_experience · noise_level · bed_comfort · cleanliness · food_quality가 1개씩이다.

### 3-3. Sentiment 오류 (5건)

5건이 모두 **정답은 `neutral`인데 모델이 평가를 덧붙인** 경우다.

| aspect | 정답 | 예측 | 구절의 성격 |
| --- | --- | --- | --- |
| stay_duration | neutral | positive | 짧게 들르기를 권하는 말 |
| noise_level | neutral | negative | 음악 소리가 크다는 사실 |
| scenery | neutral | positive | 바다가 바로 앞에 있다는 사실 |
| waiting_time | neutral | positive | 웨이팅이 있었다는 사실 |
| photo_spots | neutral | positive | 사람들이 사진을 찍는다는 관찰 |

가이드라인(평가 없이 사실만 말하면 `neutral`)과 가장 자주 어긋나는 지점이다. 다음 개선 대상으로 적합하다.

### 3-4. 그 밖

- Attribute 오류 1건: room_size `cramped`를 `average`로 예측했다.
- 허용값 위반 1건: `bed_comfort`에 `clean`을 썼다. Base도 같은 리뷰에서 같은 값을 틀렸다.
- Base에서 보이던 traveler_context 형식 오류(`{"type": "parents"}`처럼 문자열 대신 객체를 쓴 것)는 사라졌다.

## 4. 해석할 때 주의할 점

- **표본이 작은 지표:** Test에서 정답 aspect가 빈 리뷰는 4건, traveler_context가 있는 리뷰는 2건이다. 빈 리뷰 정확도와 traveler_context F1은 방향만 참고한다.
- **정밀도 조건이 다르다:** Base는 16bit, QLoRA는 4bit다. 4bit 양자화는 보통 성능을 조금 낮춘다. 그러니 이 비교의 개선 폭은 보수적으로 잡힌 쪽이다. 같은 4bit 조건의 Base 기준선은 아직 없다.
- **seed 1회:** 학습 seed 42로 한 번 돌린 결과다. seed에 따른 편차는 재지 않았다.
- **데이터 출처:** 리뷰가 부산 리뷰가 아니다. review_id 기준으로 인천 934건, 제주 213건이다. 부산 리뷰에서의 성능은 따로 확인해야 한다.
- **Teacher와의 관계:** Silver 라벨을 만든 Teacher는 Ollama `gemma4:e4b`로, 같은 E4B-it 계열이다. Base 결과는 Teacher 수준의 기준선에 가깝다고 볼 수 있다. Ollama 자체를 같은 채점기로 평가한 결과는 아직 없다.

## 5. 학습 과정

| epoch | step | 학습 loss (그 구간 마지막 기록) | 검증 loss |
| --- | ---: | ---: | ---: |
| 1 | 58 | 0.0286 | 0.0371 |
| 2 | 116 | 0.0206 | **0.0353** |
| 3 | 174 | 0.0123 | 0.0374 |

- 학습 loss는 0.171(step 10)에서 0.012까지 계속 내려갔다. 반면 검증 loss는 2 epoch에서 가장 낮고 3 epoch에서 조금 올랐다. 가벼운 과적합 신호지만 차이는 작다.
- 저장된 어댑터와 이 문서의 Test 결과는 3 epoch 끝 상태다. 2 epoch 체크포인트(`checkpoint-116`)는 artifacts에 남아 있다.
- 두 체크포인트는 **Validation으로 비교해** 고른다. Test 점수를 보고 고르면 점수가 부풀려진다 (`docs/evaluation-plan.md` 1-3절).

## 6. 효율

| 항목 | 값 |
| --- | --- |
| 학습 시간 | 83.6분 (174 step) |
| 학습 최대 VRAM | 10.95GB (reserved) / 10.72GB (allocated) |
| 학습 파라미터 | 34,881,536개 |
| 어댑터 크기 | 66.6MB |
| 추론 (QLoRA) | 리뷰당 1.98초 · 초당 29.4토큰 |

Base 추론 속도(리뷰당 11.0초)는 비교에 쓰지 않는다. 16bit 가중치(약 16GB)가 12GB VRAM에 다 들어가지 않았고, Windows 드라이버가 일반 RAM을 빌려 쓰며 느려진 값이다.

## 7. 다음 단계 후보

1. `checkpoint-116`(2 epoch)과 최종 어댑터(3 epoch)를 Validation에서 비교해 어댑터를 고른다.
2. 4bit Base 기준선을 추가해 양자화 효과와 튜닝 효과를 분리한다.
3. 16bit LoRA를 VRAM 16GB 이상 GPU에서 학습해 LoRA vs QLoRA 비교를 채운다.
4. neutral 판단 오류를 줄이는 방향으로 prompt 규칙이나 학습 데이터를 보강한다. 이때 판단은 Validation으로만 한다.
5. paired bootstrap을 평가 코드에 넣어 metrics.json에 신뢰구간을 기록한다.

## 재현

```bash
uv run python training/scripts/evaluate.py --config training/configs/base_eval.yaml
uv run python training/scripts/train.py --config training/configs/qlora_gold_v1.yaml
uv run python training/scripts/compare_runs.py training/experiments/base_v001 training/experiments/qlora_gold_v001
```

CUDA torch를 따로 설치했다면 `UV_NO_SYNC=1`을 설정한다 (`training/README.md` 4절).
