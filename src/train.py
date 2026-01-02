import os
import json
import argparse
import pandas as pd
from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
import numpy as np



# ============================================
# 1. Argument Parser (epoch, lr, batch Size)
# ============================================

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--batch", type=int, default=8)
args = parser.parse_args()

EPOCHS = args.epochs
LR = args.lr
BATCH = args.batch

## ============================================
# 2. Load Data
# ============================================

df = pd.read_csv("/content/drive/MyDrive/toefl/data/processed/output_pos_tagged.csv")

TEXT_COLUMN = "content"
POS_COLUMN = "content_pos"

# 🔥 content + POS 결합
df["text_with_pos"] = (
    df[TEXT_COLUMN].astype(str)
    + " [SEP] "
    + df[POS_COLUMN].astype(str)
)

le = LabelEncoder()
df["label"] = le.fit_transform(df["level"])

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]   # 🔥 accuracy 안정화
)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ============================================
# 3. Tokenizer
# ============================================

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text_with_pos"],   # 🔥 여기 변경
        truncation=True,
        padding="max_length",
        max_length=256             # POS 때문에 384도 실험 가치 있음
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)
test_dataset.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(le.classes_)
)


# ============================================
# 0. Auto Experiment Folder
# ============================================

exp_root = "/content/drive/MyDrive/toefl/output/reports/exp003_pos_epoch20"
os.makedirs(exp_root, exist_ok=True)

existing = sorted([d for d in os.listdir(exp_root) if d.startswith("exp")])
if existing:
    last_num = int(existing[-1].split("_")[0].replace("exp", ""))
else:
    last_num = 0

exp_num = last_num + 1
exp_name = f"exp{exp_num:03d}_epoch{EPOCHS}"
save_dir = os.path.join(exp_root, exp_name)
os.makedirs(save_dir, exist_ok=True)

print(f"🔹 생성된 실험 폴더: {save_dir}")


training_args = TrainingArguments(
    output_dir=save_dir,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    learning_rate=LR,
    logging_dir=os.path.join(save_dir, "logs"),
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    
)
# ============================================ # 9. Train model # ============================================ 
trainer.train() 

# ============================================ # 10. Save Model + Tokenizer # ============================================ 
model.save_pretrained(save_dir) 
tokenizer.save_pretrained(save_dir) 

print("✔ 모델 가중치 및 토크나이저 저장 완료") 

# ============================================ # 11. Predict & Save # ============================================ 
predictions = trainer.predict(test_dataset) 
pred_labels = predictions.predictions.argmax(axis=1) 
test_df["y_pred"] = le.inverse_transform(pred_labels) 
test_df["y_true"] = le.inverse_transform(test_df["label"]) 
pred_path = os.path.join(save_dir, "prediction_results.csv") 
test_df[["y_true", "y_pred"]].to_csv(pred_path, index=False) 
print(f"🎉 예측 결과 저장 완료: {pred_path}") 
print("🎯 실험 완료!")