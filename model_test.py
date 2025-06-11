from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
import os

# 載入 tokenizer、model、label encoder
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "finetune_output")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

# 預測句子
text = "這台機器無法開機"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    print("分類結果：", label_encoder.inverse_transform([pred])[0])