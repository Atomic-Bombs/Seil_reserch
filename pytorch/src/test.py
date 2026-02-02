import torch
import matplotlib.pyplot as plt
import numpy as np

# ランダムな画像データ (3x256x256) を生成
# これは 3 チャンネル (RGB)、256x256 の画像
image = torch.rand(3, 256, 256)

# 画像の正規化 (0-1 -> -1 から 1 の範囲に変換)
normalized_image = 2 * image - 1  # [-1, 1] にスケーリング

# 正規化後の画像を numpy 配列に変換して matplotlib で表示
normalized_image_np = normalized_image.numpy()

# 画像を表示
# 軸の順番を (H, W, C) にする必要があるので、(C, H, W) -> (H, W, C) に変換
plt.imshow(np.transpose(normalized_image_np, (1, 2, 0)))
plt.title("Normalized Image")
plt.axis("off")  # 軸を表示しない

# バックエンドを 'Agg' に設定して画像を保存
plt.savefig("normalized_image.png", bbox_inches='tight', pad_inches=0)
plt.close()  # 表示せずにファイルに保存

# 'normalized_image.png' として保存されます
