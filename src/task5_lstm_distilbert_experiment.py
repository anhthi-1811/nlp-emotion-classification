import os
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from utils import set_seed
from data_pipeline import DataPipelineManager

# 0. KHỞI TẠO & CỐ ĐỊNH SEED
set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Đang chạy trên thiết bị: {device}")


# 1. DATA PIPELINE (Kết nối với code của bạn)
# Tải mock_data.csv
pipeline = DataPipelineManager(raw_data_path="mock_data.csv", mock_data_output_path="mock_data.csv")
df_mock = pipeline.load_raw_data()

# Tạm thời chia train/val (80/20) trực tiếp trên mock_data
df_train, df_val = train_test_split(df_mock, test_size=0.2, random_state=42, stratify=df_mock['label'])

# Khởi tạo DataLoader
train_loader = pipeline.get_pytorch_loaders(df_train, batch_size=16, shuffle=True)
val_loader = pipeline.get_pytorch_loaders(df_val, batch_size=32, shuffle=False)

# Số lượng nhãn (classes) và Tokenizer
NUM_CLASSES = df_mock['label'].nunique()
TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


# 2. XÂY DỰNG KIẾN TRÚC MÔ HÌNH (Task 5)

# --- MÔ HÌNH 1: BASELINE LSTM ---
class LSTM_Baseline(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(LSTM_Baseline, self).__init__()
        # Tạo lớp nhúng từ vựng dựa trên kích thước của Tokenizer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Sử dụng LSTM 2 chiều (Bidirectional) với 1 tầng mạng để làm baseline nhẹ gọn
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=1, bidirectional=True)
        # Lớp phân loại đầu ra (hidden_dim * 2 vì là LSTM 2 chiều)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        # Kỹ thuật lấy trạng thái (state) cuối cùng của chuỗi:
        # Chiều tiến (Forward) lấy ở từ cuối cùng (index -1), chiều lùi (Backward) lấy ở từ đầu tiên (index 0)
        hidden_dim = self.lstm.hidden_size
        out = torch.cat((lstm_out[:, -1, :hidden_dim], lstm_out[:, 0, hidden_dim:]), dim=-1)
        return self.fc(out)


# --- MÔ HÌNH 2: DISTILBERT NÂNG CAO ---
class DistilBert_Advanced(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super(DistilBert_Advanced, self).__init__()
        # Tải kiến trúc gốc pre-trained từ HuggingFace
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # # [Task 6] Layer Freezing: Đóng băng lớp dưới
        # if freeze_backbone:
        #     for name, param in self.distilbert.named_parameters():
        #         # Chỉ mở khóa layer transformer cuối cùng (layer 5) và pooler
        #         if "transformer.layer.5" not in name:
        #             param.requires_grad = False

        # Lớp phân loại sắc thái đầu ra bọc ngoài backbone
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        # Trích xuất tầng ẩn cuối cùng của token đặc biệt [CLS] đại diện cho ngữ nghĩa toàn câu
        cls_output = outputs.last_hidden_state[:, 0, :] # Lấy [CLS] token
        return self.classifier(cls_output)


# 3. VIẾT TRAINING LOOP CHUẨN & LINH HOẠT
def run_task5_epoch(model_type="DistilBERT"):
    # Cấu hình siêu tham số cơ bản
    config = {
        "architecture": model_type,
        "learning_rate": 2e-5 if model_type == "DistilBERT" else 1e-3,
        "epochs": 1, # Chỉ chạy thử 1 epoch
        "batch_size": 32 if model_type == "LSTM" else 16
    }

    # Kết nối và khai báo thông tin thực nghiệm lên W&B
    wandb.init(
        entity="nlp-emotion",
        project="nlp-emotion-classification",
        name=f"{model_type}_Task5_Test",
        config=config
    )

    # Khởi tạo mô hình tương ứng dựa trên cấu hình lựa chọn linh hoạt
    if model_type == "DistilBERT":
        model = DistilBert_Advanced(NUM_CLASSES).to(device)
    elif model_type == "LSTM":
        model = LSTM_Baseline(TOKENIZER.vocab_size, 128, 128, NUM_CLASSES).to(device)
    else:
        raise ValueError("Vui lòng chọn chính xác 'LSTM' hoặc 'DistilBERT'")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    print(f"\n [Task 5] Bắt đầu chạy thử 1 Epoch với mô hình: {model_type}...")

    # --- PHASE: TRAINING ---
    model.train()
    train_loss = 0

    for texts, labels in tqdm(train_loader, desc="Huấn luyện"):
        # Tokenize chuỗi văn bản thô trực tiếp trong loop do data_pipeline chưa xử lý văn bản
        inputs = TOKENIZER(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Lan truyền tiến linh hoạt cho cả 2 kiểu kiến trúc mạng
        if isinstance(model, LSTM_Baseline):
            outputs = model(input_ids)
        else:
            outputs = model(input_ids, attention_mask)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # --- PHASE: VALIDATION ---
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for texts, labels in val_loader:
            inputs = TOKENIZER(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device)

            if isinstance(model, LSTM_Baseline):
                outputs = model(input_ids)
            else:
                outputs = model(input_ids, attention_mask)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Kết quả 1 Epoch | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1-score: {val_f1:.4f}")

    # Đẩy log chỉ số đo đạc thực tế lên dashboard hệ thống W&B
    wandb.log({
        "epoch": 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_f1_score": val_f1
    })

    wandb.finish()



# 5. THỰC THI KIỂM TRA
if __name__ == "__main__":
    run_task5_epoch(model_type="LSTM")
    run_task5_epoch(model_type="DistilBERT")
