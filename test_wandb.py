import wandb
import random
import time

# 1. Khởi tạo W&B (Khai báo tên dự án và cấu hình)
wandb.init(
    entity="nlp-emotion",
    project="nlp-emotion-classification", 
    name="Bi-LSTM_Test_Run", 
    config={
        "architecture": "Bi-LSTM",
        "learning_rate": 0.001,
        "epochs": 5,
        "batch_size": 32
    }
)
# # Khởi tạo W&B cho DistilBERT 
# wandb.init(
#     entity="nlp-emotion",
#     project="nlp-emotion-classification",  # GIỮ NGUYÊN: Để vào chung một dự án
#     name="DistilBERT_Test_Run",  # THAY ĐỔI: Tên lần chạy của DistilBERT
#     config={
#         "architecture": "DistilBERT",  # THAY ĐỔI: Tên kiến trúc mô hình
#         "learning_rate": 2e-5,  # THAY ĐỔI: Thường DistilBERT dùng lr nhỏ hơn LSTM
#         "epochs": 5,
#         "batch_size": 16,  # THAY ĐỔI: Model nặng nên thường giảm batch_size
#     },
# )

print("Bắt đầu huấn luyện mô hình giả lập...")

# 2. Vòng lặp huấn luyện (Giả lập)
epochs = wandb.config.epochs
for epoch in range(epochs):
    # Tạo ra các con số ảo (F1 tăng dần, Loss giảm dần) để test
    train_loss = 0.8 - (epoch * 0.1) + random.uniform(-0.05, 0.05)
    val_f1 = 0.5 + (epoch * 0.08) + random.uniform(-0.02, 0.02)
    
    # BẮT BUỘC: Gửi dữ liệu lên dashboard W&B
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_f1_score": val_f1
    })
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | F1: {val_f1:.4f}")
    time.sleep(1) # Dừng 1 giây cho giống đang train thật

# 3. Đóng kết nối
wandb.finish()
print("Hoàn tất! Hãy lên trang chủ wandb.ai để xem biểu đồ.")