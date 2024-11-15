import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# 設置路徑
root_path = os.path.abspath(os.path.dirname(__file__))

# 超參數設置
learning_rate = 0.0001
epochs = 50
batch_size = 32

def load_training_data():
    # 加載訓練數據
    train_dataset = np.load(os.path.join(root_path, 'dataset', 'train.npz'))
    train_data = train_dataset['data']
    train_label = to_categorical(train_dataset['label'])  # 使用 one-hot 編碼
    return train_data, train_label

def load_validation_data():
    # 加載驗證數據
    valid_dataset = np.load(os.path.join(root_path, 'dataset', 'validation.npz'))
    valid_data = valid_dataset['data']
    valid_label = to_categorical(valid_dataset['label'])  # 使用 one-hot 編碼
    return valid_data, valid_label

def train_model():
    # 定義模型，移除 batch_shape，改為使用 input_shape
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(25,)),  # 假設輸入形狀為 (25,)
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(5, activation='softmax')  # 假設有 5 個分類
    ])
    
    # 編譯模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # 加載訓練數據
    train_data, train_label = load_training_data()

    # 訓練模型
    model.fit(train_data, train_label, batch_size=batch_size, epochs=epochs)

    # 保存模型
    model.save(os.path.join(root_path, 'YOURMODEL.h5'))

def evaluate_model():
    # 加載訓練好的模型
    model = tf.keras.models.load_model(os.path.join(root_path, 'YOURMODEL.h5'))

    # 加載驗證數據
    valid_data, valid_label = load_validation_data()

    # 進行預測
    predictions = model.predict(valid_data, batch_size=batch_size)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(valid_label, axis=1)

    # 計算準確率
    accuracy = np.mean(true_labels == predicted_labels)
    print(f'Predicted labels: {predicted_labels}')
    print(f'True labels: {true_labels}')
    print(f'Accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    # 訓練模型
    train_model()

    # 評估模型
    evaluate_model()
