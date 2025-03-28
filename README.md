# 📈 LSTM 기반 트래픽 예측 시스템

이 프로젝트는 시간 단위 트래픽 데이터를 기반으로 LSTM 모델을 학습하여, 미래의 트래픽을 예측하고 Streamlit 대시보드를 통해 결과를 시각화하며, Google Cloud Storage(GCS)에 예측 결과를 업로드할 수 있는 기능까지 제공합니다.

---

## 🔧 프로젝트 구조

```
traffic_predictor/
├── app/
│   └── streamlit_app.py         # Streamlit 대시보드 실행 파일
├── data/
│   └── Midterm_53_group.csv     # 원본 트래픽 데이터
├── lstm_model/
│   ├── model.py                 # LSTM 모델 클래스 정의
│   └── dataset.py               # 학습용 시퀀스 생성 Dataset 클래스
├── .gitignore
└── README.md
```

---

## 🚀 기능 요약

### ✅ 트래픽 데이터 전처리
- Timestamp → datetime 변환
- 분(minute) 단위로 트래픽 합산
- 정규화(min-max scaling)

### ✅ LSTM 예측 모델
- PyTorch 기반 LSTM 모델 학습 (train.py)
- 예측 결과 추론 및 시각화

### ✅ Streamlit 대시보드
- 최근 트래픽 시각화
- 1분 후 예측값 출력
- 버튼 클릭으로 GCS 업로드

### ✅ GCS 연동
- `google-cloud-storage` 라이브러리 사용
- `anomaly_result.csv`를 버킷에 업로드

---

## 🛠 사용법

### 1. 가상환경 생성 및 라이브러리 설치
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2. Streamlit 실행
```bash
cd traffic_predictor
streamlit run app/streamlit_app.py
```

### 3. GCP 인증 설정
```bash
# GCP 서비스 계정 키 환경변수 등록
set GOOGLE_APPLICATION_CREDENTIALS=경로\to\your-key.json
```

---

## 📦 주요 패키지
- `torch`
- `pandas`, `numpy`
- `streamlit`
- `google-cloud-storage`
- `matplotlib`

---

## ✨ TODO / 개선 사항
- 예측 구간 확대 (multi-step prediction)
- 실시간 예측 스케줄링 (Cloud Functions 등 연동)
- 이상 탐지 기능 추가
- Streamlit 업로드 이력 표시

---

## 📄 라이선스
MIT License

