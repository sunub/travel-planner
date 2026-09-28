# training/experiments

Git으로 관리하는 실험 기록이다. run 하나마다 폴더 하나가 생긴다 (`<experiment_name>_v001`, `_v002`, ...).

```
<run>/
  config.yaml        실행에 쓴 최종 설정 (defaults와 --set을 모두 반영한 것)
  metrics.json       Base / LoRA / QLoRA가 모두 같은 모양으로 쓰는 결과
  summary.md         사람이 읽는 요약표
  loss_history.csv   step별 train loss / eval loss (학습 run만)
```

어댑터, 체크포인트, 예측 결과처럼 리뷰 원문이 담기거나 큰 파일은 `output.artifacts_root`(기본 `training/artifacts/`, Git 제외)에 저장된다.
