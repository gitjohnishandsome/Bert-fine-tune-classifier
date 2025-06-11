from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
import os
import pandas as pd

# 載入 tokenizer、model、label encoder
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "finetune_output_v2")
test_path = os.path.join(base_dir, "testdata500_0605.csv")
ori_training_data_path = os.path.join(base_dir, "fine_tune_dataset_v4.csv")
last_training_data_path = os.path.join(base_dir, "fine_tune_dataset_v5.csv")

output_path = os.path.join(base_dir, "prediction_result_3.xlsx")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

# 嘗試讀取測試資料
try:
    df = pd.read_csv(test_path, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(test_path, encoding="cp950")

df = df.dropna(subset=["text", "label"])  # 移除 text 或 label 是空值的列
df = df.reset_index(drop=True)  # 重新編號，避免 index 出錯
#df["text"] = df["text"].astype(str)
# 儲存預測與比對結果
predicted_labels = []
match_results = []

total = len(df)
correct = 0

for idx, row in df.iterrows():
    inputs = tokenizer(row['text'], return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        predicted_label = label_encoder.inverse_transform([pred])[0]

    predicted_labels.append(predicted_label)

    if predicted_label == row["label"]:
        correct += 1
        match_results.append("✔ 正確")
    else:
        match_results.append("✘ 錯誤")

# 計算準確率
accuracy = correct / total
print(f"✅ 測試資料準確率：{accuracy:.2%}")

# 加入預測欄位
df["預測結果"] = predicted_labels
df["配對結果"] = match_results

error_df = df[df["配對結果"] == "✘ 錯誤"]
error_output_path = os.path.join(base_dir, "errors_for_finetune.csv")
error_df[["text", "label"]].to_csv(error_output_path, index=False, encoding="utf-8-sig")
# 輸出錯誤資料
print(f"❌ 錯誤資料已儲存至：{error_output_path}")
# 合併原始資料與錯誤資料

try:
    df_original = pd.read_csv(ori_training_data_path, encoding="utf-8")
    print("✅ 成功以utf-8讀取df_original")
except UnicodeDecodeError:
    df_original = pd.read_csv(ori_training_data_path, encoding="cp950")
    print("✅ 成功以cp950讀取df_original")

try:
    df_errors = pd.read_csv(error_output_path, encoding="utf-8")
    print("✅ 成功以utf-8讀取df_errors")
except UnicodeDecodeError:
    df_errors = pd.read_csv(error_output_path, encoding="cp950")
    print("✅ 成功以cp950讀取df_errors")

# 編碼一致處理（建議保留）
df_original["text"] = df_original["text"].astype(str)
df_errors["text"] = df_errors["text"].astype(str)

# 只保留在錯誤資料中「不在原始資料中的項目」
df_new = df_errors.merge(df_original, on=["text", "label"], how="left", indicator=True)
df_new = df_new[df_new["_merge"] == "left_only"].drop(columns=["_merge"])

# 合併資料
df_combined = pd.concat([df_original, df_new], ignore_index=True)
df_combined.to_csv(last_training_data_path, index=False, encoding="utf-8-sig")
print(f"📁 已將結果儲存至：{last_training_data_path}")
# 儲存為 Excel
df.to_excel(output_path, index=False)
print(f"📁 已將結果儲存至：{output_path}")





