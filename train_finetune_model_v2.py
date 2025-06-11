from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import os

# ======== 路徑設定 ========
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "fine_tune_dataset_v2.csv")
output_dir = os.path.join(base_dir, "finetune_output_v2")
pkl_path = os.path.join(output_dir, "label_encoder.pkl")
os.makedirs(output_dir, exist_ok=True)

# ======== 偵測 CSV 編碼並讀取 ========
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw = f.read()
    for enc in ["utf-8", "cp950", "big5"]:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    raise ValueError("無法辨識檔案編碼格式")

encoding = detect_encoding(csv_path)
df = pd.read_csv(csv_path, encoding=encoding)

# ======== 編碼 Label 並拆分訓練集/驗證集 ========
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])
df["label_id"] = df["label_id"].astype(int)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# ======== tokenizer 預處理 ========
model_checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

train_dataset = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "label"}))
val_dataset = Dataset.from_pandas(val_df[["text", "label_id"]].rename(columns={"label_id": "label"}))

tokenized_train = train_dataset.map(preprocess)
tokenized_val = val_dataset.map(preprocess)

# ======== 模型初始化 ========
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_encoder.classes_)
)

# ======== 訓練參數與指標 ========
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    logging_dir=os.path.join(output_dir, "logs"),
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics
)

# ======== 開始訓練 ========
trainer.train()

# ======== 儲存模型、tokenizer、label encoder ========
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
joblib.dump(label_encoder, pkl_path)

print("✅ 模型訓練與驗證完成！")
