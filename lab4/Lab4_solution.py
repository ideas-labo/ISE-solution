import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


# 1. 加载和预处理数据
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # 打印列名，检查目标列
    print("Columns in dataset:", df.columns)

    target_column = 'income' # 根据输出列名修改目标列名
    X = df.drop(columns=[target_column])  # 请根据实际的目标列名替换 'target'
    y = df[target_column]

    # 对类别特征进行数值化
    label_encoder = LabelEncoder()
    X = X.apply(label_encoder.fit_transform)
    y = label_encoder.fit_transform(y)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


# 2. 定义 6层全连接 DNN 模型
def build_dnn_model(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))  # Dropout 防止过拟合
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 二分类问题，sigmoid 激活函数

    # 使用 Adam 优化器
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


# 3. 训练和评估模型
file_path = 'processed_kdd_cleaned.csv'  # 数据集路径

# 1. 加载数据集与预处理
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

# 2. 创建 6层 DNN 模型
model = build_dnn_model(X_train.shape[1], learning_rate=0.001)

# 3. 训练模型
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# 4. 保存模型
model_save_path = f'model_{file_path.split("/")[-1].split(".")[0]}.h5'
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# 5. 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Model accuracy: {accuracy:.4f}')
