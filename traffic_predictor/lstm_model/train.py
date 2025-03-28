import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from dataset import TrafficDataset
from model import LSTMRegressor

# parameter
INPUT_SIZE = 10
PRED_SIZE = 1
BATCH_SIZE = 4
EPOCHS = 20
LR = 0.001

# Midterm 데이터 불러오기
csv_path = os.path.join("..", "data", "Midterm_53_group.csv")
df = pd.read_csv(csv_path)

#  Time → datetime 변환
start_time = pd.to_datetime("2023-01-01 00:00:00")
df['datetime'] = df['Time'].apply(lambda x: start_time + pd.to_timedelta(x, unit='s'))

#  분 단위로 트래픽 합산
df_agg = df.resample('min', on='datetime')['Length'].sum().reset_index()

df_agg.rename(columns={'Length': 'traffic'}, inplace=True)

# 시계열 추출
series = df_agg['traffic'].values
print("시계열 길이:", len(series))
min_val = series.min()
max_val = series.max()
series = (series - min_val) / (max_val - min_val)
#정규화
#  Dataset, DataLoader
dataset = TrafficDataset(series, input_size=INPUT_SIZE, pred_size=PRED_SIZE)
print("시퀀스 개수:", len(dataset))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#  모델 정의
model = LSTMRegressor(input_size=1, hidden_size=64, num_layers=2, output_size=PRED_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#  손실함수 & 옵티마이저
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 학습 루프
losses = []
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for x, y in dataloader:
        x = x.unsqueeze(-1).to(device)  # (batch, seq_len, 1)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"[{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

# 손실 그래프
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()

#  모델 저장
torch.save(model.state_dict(), "lstm_model.pth")