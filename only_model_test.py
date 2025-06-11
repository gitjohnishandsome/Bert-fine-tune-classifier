from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
import pandas as pd
import os

# 設定路徑
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "finetune_output_v2")  # 微調後模型資料夾
test_path = os.path.join(base_dir, "test_data_500.csv")  # 測試資料路徑
output_path = os.path.join(model_dir, "prediction_results.xlsx")  # 預測輸出檔案

# 載入模型與編碼器
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

# 讀取測試資料
df = pd.read_csv(test_path, encoding="utf-8")

# 初始化計數器與結果列表
results = []
correct = 0
total = len(df)

# 預測每一筆資料
for idx, row in df.iterrows():
    text = str(row["text"])
    label_true = row["label"]

    # 文字轉向量
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        label_pred = label_encoder.inverse_transform([pred])[0]

    # 是否正確
    is_correct = label_pred == label_true
    if is_correct:
        correct += 1

    # 存結果
    results.append({
        "text": text,
        "true_label": label_true,
        "predicted_label": label_pred,
        "match": "✅" if is_correct else "❌"
    })

# 準確率
accuracy = correct / total
print(f"✅ 測試資料準確率：{accuracy:.2%}")

# 存成 Excel
result_df = pd.DataFrame(results)
result_df.to_excel(output_path, index=False)  # ✅ 刪除 encoding 參數
print(f"📄 預測結果已儲存至：{output_path}")