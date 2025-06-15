from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import pandas as pd
import joblib
import os

# 路徑設定
base_dir = os.path.dirname(os.path.abspath(__file__))# 獲取當前檔案所在目 錄

csv_path = os.path.join(base_dir, "fine_tune_dataset_v5.csv")# CSV 檔案路徑，用來讀取訓練資料
output_dir = os.path.join(base_dir, "finetune_output_v2")# 訓練完模型輸出的資料夾位置
pkl_path = os.path.join(output_dir, "label_encoder.pkl")# 訓練完模型儲存 label encoder 的路徑
os.makedirs(output_dir, exist_ok=True)

# 自動偵測編碼並讀取 CSV
def detect_encoding(file_path):# 自動偵測 CSV 檔案(訓練資料)的文字編碼格式
    with open(file_path, "rb") as f:
        raw = f.read()
    for enc in ["utf-8", "cp950", "big5"]:
        try:
            raw.decode(enc)#raw.decode(enc) 嘗試將原始資料用指定編碼「轉換成文字」
            return enc
        except UnicodeDecodeError:
            continue
    raise ValueError("無法辨識檔案編碼格式")

encoding = detect_encoding(csv_path)# 偵測訓練檔案的編碼格式
df = pd.read_csv(csv_path, encoding=encoding)# 讀取 CSV 檔案，並使用偵測到的編碼格式

# 編碼 label
label_encoder = LabelEncoder()# 把每個label賦予一個對應的「整數代號」(0，1，2，3......)
df["label_id"] = label_encoder.fit_transform(df["label"])
# csv檔往右新增一行(label_id)把分類任務的文字標籤賦予數值， BERT 只能輸出數字類型的分類。
df["label_id"] = df["label_id"].astype(int)

# 建立 Dataset 並進行 tokenizer 處理
model_checkpoint = "ckiplab/bert-base-chinese"# 使用 BERT 中文模型作為基礎模型
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess(example):# 定義預處理函數，將文本轉換為模型可接受的格式(字典)
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)




#----------這段程式碼的功能是訓練一個 BERT 模型來進行文本分類任務。----------





dataset = Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "label"}))#將訓練資料的文本和標籤轉換為 Hugging Face 的 Dataset 格式，並重命名標籤欄位為 "label"
tokenized_dataset = dataset.map(preprocess)# 將文本資料進行預處理，轉換為模型可接受的格式

# 模型初始化
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_encoder.classes_)
)

# 訓練參數
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    logging_dir=os.path.join(output_dir, "logs"),
    save_strategy="epoch",
    logging_strategy="steps",           # 每幾步記錄一次
    logging_steps=10,                   # 每 10 steps 顯示一次 loss
    report_to="none"                    # 不上傳到 wandb 等外部工具
)

# 建立 Trainer 並訓練
trainer = Trainer(
    model=model,
    args=training_args,#訓練參數
    train_dataset=tokenized_dataset,# 訓練資料集
)

trainer.train()# 開始訓練模型

# 儲存模型、tokenizer、label encoder
model.save_pretrained(output_dir)#儲存模型到指定路徑
tokenizer.save_pretrained(output_dir)# 儲存 tokenizer 到指定路徑
joblib.dump(label_encoder, pkl_path)# 儲存 label encoder 到指定路徑

print("✅ 模型訓練與儲存完成！")
