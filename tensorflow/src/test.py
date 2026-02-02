import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# 1. MNISTデータセットをロード
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. データの前処理
# 画像データを0-1に正規化
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

# 3. データの形状をTensorFlow用に変更
X_train_scaled = X_train_scaled.reshape(-1, 28, 28, 1)
X_test_scaled = X_test_scaled.reshape(-1, 28, 28, 1)

# 4. 畳み込みニューラルネットワーク(CNN)モデルを作成
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),  # 最初の一行にこれを入れる
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 5. モデルのコンパイル
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 6. モデルの訓練
model.fit(X_train_scaled, y_train, epochs=5, batch_size=64, verbose=1)

# 7. テストデータでモデルを評価
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=2)
print(f'Test accuracy: {test_acc:.4f}')

# 8. 予測を行い、最初の5つの画像を表示
y_pred = model.predict(X_test_scaled[:5])

# 9. 結果の可視化
# 'agg' バックエンドを設定して画像をac保存する
plt.switch_backend('agg')  # 'agg' バックエンドに変更
fig, axes = plt.subplots(1, 5, figsize=(100, 3))
for i, ax in enumerate(axes):
    ax.imshow(X_test_scaled[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Pred: {np.argmax(y_pred[i])} | True: {y_test[i]}')
    ax.axis('off')

# 画像をファイルに保存
plt.savefig('mnist_predictions.png', bbox_inches='tight', pad_inches=0)
plt.close()

# 'mnist_predictions.png' として画像が保存されます
