import wandb
import random
import time

# 1. Initialize W&B (Declare project name and config)
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

# # Initialize W&B for DistilBERT 
# wandb.init(
#     entity="nlp-emotion",
#     project="nlp-emotion-classification",  # KEEP THIS: To stay in the same project workspace
#     name="DistilBERT_Test_Run",  # CHANGE: Run name for DistilBERT
#     config={
#         "architecture": "DistilBERT",  # CHANGE: Model architecture name
#         "learning_rate": 2e-5,  # CHANGE: DistilBERT usually requires a smaller lr than LSTM
#         "epochs": 5,
#         "batch_size": 16,  # CHANGE: Heavy model so batch_size is usually reduced
#     },
# )

print("Starting mock model training...")

# 2. Training loop (Mock)
epochs = wandb.config.epochs
for epoch in range(epochs):
    # Generate mock numbers (F1 increasing, Loss decreasing) for testing
    train_loss = 0.8 - (epoch * 0.1) + random.uniform(-0.05, 0.05)
    val_f1 = 0.5 + (epoch * 0.08) + random.uniform(-0.02, 0.02)
    
    # REQUIRED: Send data to W&B dashboard
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_f1_score": val_f1
    })
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | F1: {val_f1:.4f}")
    time.sleep(1) # Pause for 1 second to simulate real training time

# 3. Close connection
wandb.finish()
print("Done! Check your W&B dashboard at wandb.ai to see the charts.")
