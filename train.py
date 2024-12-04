import numpy as np
import pickle
from data import load_data
from model import CNNModel

# 데이터 로드
(train_images, train_labels), (test_images, test_labels) = load_data('.')

# 하이퍼파라미터 설정
learning_rate = 0.01
batch_size = 64
epochs = 5
num_batches = len(train_images) // batch_size

# 모델 초기화
model = CNNModel(learning_rate=learning_rate)

# Softmax 함수 정의
def softmax(input):
    max_val = np.max(input)  # Stability를 위한 최대값 제거
    exp_vals = np.exp(input - max_val)
    probabilities = exp_vals / np.sum(exp_vals)
    return probabilities

# 손실 함수 정의
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-6))

# 학습 루프
for epoch in range(epochs):
    epoch_loss = 0
    for batch_idx in range(num_batches):
        # 배치 데이터 로드
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        batch_images = train_images[batch_start:batch_end]
        batch_labels = train_labels[batch_start:batch_end]

        batch_loss = 0
        for i in range(batch_size):
            # 입력 데이터에 채널 차원 추가
            input_data = batch_images[i].reshape(1, 28, 28)  # (1, 28, 28)로 reshape

            # Forward Pass
            y_pred = model.forward(input_data)
            y_pred = softmax(y_pred)

            # Loss 계산
            loss = cross_entropy_loss(batch_labels[i], y_pred)
            batch_loss += loss

            # Backward Pass
            
            grad = y_pred - batch_labels[i].reshape(-1, 1)  # (10, 1)로 reshape
            model.backward(grad)
        # 파라미터 업데이트
        model.update()
        epoch_loss += batch_loss / batch_size

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / num_batches:.4f}")

# 학습된 모델 저장
with open('ckpt.pkl', 'wb') as f:
    pickle.dump(model, f)
print("모델이 저장되었습니다!")
