import pandas as pd
import torch
from model import LSTMRegressor
from dataset import TrafficDataset
import matplotlib.pyplot as plt
import os

#  설정
INPUT_SIZE = 10
PRED_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드 및 전처리
csv_path = os.path.join("..", "data", "Midterm_53_group.csv")
df = pd.read_csv(csv_path)
# datetime 변환
start_time = pd.to_datetime("2023-01-01 00:00:00")
df['datetime'] = df['Time'].apply(lambda x: start_time + pd.to_timedelta(x, unit='s'))

# 분 단위로 트래픽 집계
df_agg = df.resample('min', on='datetime')['Length'].sum().reset_index()
df_agg.rename(columns={'Length': 'traffic'}, inplace=True)

# 정규화
series = df_agg['traffic'].values
min_val = series.min()
max_val = series.max()
normalized = (series - min_val) / (max_val - min_val)

#  입력 시퀀스 (가장 최근 INPUT_SIZE개)
input_seq = normalized[-INPUT_SIZE:]
input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)

#  모델 불러오기
model = LSTMRegressor(input_size=1, hidden_size=64, num_layers=2, output_size=PRED_SIZE)
model.load_state_dict(torch.load("lstm_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

#  예측
with torch.no_grad():
    predicted = model(input_tensor).cpu().numpy().flatten()

# 역정규화
predicted_denorm = predicted[0] * (max_val - min_val) + min_val

#  시각화
plt.figure(figsize=(10, 5))
plt.plot(range(INPUT_SIZE), series[-INPUT_SIZE:], label="recent traffic")
plt.plot(INPUT_SIZE, predicted_denorm, 'ro', label="prediction value (after 1 min)")
plt.axvline(INPUT_SIZE, color='gray', linestyle='--')
plt.title("LSTM traffic prediction")
plt.xlabel("time(minute)")
plt.ylabel("traffic (Length)")
plt.legend()
plt.grid(True)
plt.show()