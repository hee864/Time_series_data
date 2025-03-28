from google.cloud import storage
import os
import sys
import torch
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

#  경로 설정 및 모듈 임포트
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lstm_model')))
from model import LSTMRegressor

#  디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 10
PRED_SIZE = 1

#  파일 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "..", "data", "Midterm_53_group.csv")
model_path = os.path.join(base_dir, "..", "lstm_model", "lstm_model.pth")

# 데이터 로드 및 전처리
df = pd.read_csv(csv_path)
start_time = pd.to_datetime("2023-01-01 00:00:00")
df['datetime'] = df['Time'].apply(lambda x: start_time + pd.to_timedelta(x, unit='s'))
df_agg = df.resample('min', on='datetime')['Length'].sum().reset_index()
df_agg.rename(columns={'Length': 'traffic'}, inplace=True)

# 정규화
series = df_agg['traffic'].values
min_val = series.min()
max_val = series.max()
normalized = (series - min_val) / (max_val - min_val)

# 입력 시퀀스 준비
input_seq = normalized[-INPUT_SIZE:]
input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)

#  모델 로드 및 예측
model = LSTMRegressor(input_size=1, hidden_size=64, num_layers=2, output_size=PRED_SIZE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

with torch.no_grad():
    predicted = model(input_tensor).cpu().numpy().flatten()
predicted_denorm = predicted[0] * (max_val - min_val) + min_val

#  Streamlit UI
st.title("LSTM 트래픽 예측 대시보드")
st.line_chart(pd.Series(series[-INPUT_SIZE:], name="최근 트래픽"))
st.write(f"### 현재 예측값 (1분 후): {predicted_denorm:,.0f} Bytes")

# 최근 트래픽
recent_values = list(series[-INPUT_SIZE:])
# 예측값 추가용
recent_values_with_prediction = recent_values + [None]
predicted_values = [None] * INPUT_SIZE + [predicted_denorm]
#  예측 결과 저장용 DataFrame 생성

result_df = pd.DataFrame({
    "minute": list(range(-INPUT_SIZE + 1, 1)) + [1],
    "traffic": recent_values_with_prediction,
    "predicted": predicted_values
})

#  GCP 업로드 함수 정의
def upload_to_gcs(local_file, bucket_name, destination_blob_name):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file)
        return True
    except Exception as e:
        st.error(f"GCP 업로드 실패: {e}")
        return False

#  GCP 업로드 버튼
if st.button("결과 GCP에 업로드"):
    result_df.to_csv("anomaly_result.csv", index=False)
    st.success("CSV 저장 완료!")

    uploaded = upload_to_gcs(
        local_file="anomaly_result.csv",
        bucket_name="my-traffic-bucket",  # ← 실제 GCS 버킷 이름으로 변경
        destination_blob_name="streamlit/anomaly_result.csv"  # GCS 내부 경로
    )

    if uploaded:
        st.success("GCP 업로드 성공!")