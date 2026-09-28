"""Gemma4 LoRA/QLoRA 파인튜닝 실험 코드 (데이터 분할 · SFT 변환 · 학습 · 추론 · 평가 · 실험 기록).

data/, evaluation/, utils/ 는 torch 없이 돌아간다. training/, inference/ 만 GPU 라이브러리를 쓴다.
실행 진입점과 설정 파일은 저장소의 training/ 폴더에 있다.
"""
