from sklearn.preprocessing import LabelEncoder

# 建立一個 label encoder
label_encoder = LabelEncoder()

# 假設我們有三種分類
labels = ["電源問題", "螢幕故障", "無法連線", "電源問"]

# 將文字標籤轉成數字（適合訓練用）
encoded = label_encoder.fit_transform(labels)
print(encoded)
# 輸出：array([1, 2, 0, 1])