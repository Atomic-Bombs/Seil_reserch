import torch

# PyTorch の動作確認
x = torch.tensor([1.0, 2.0, 3.0])
y = x ** 2
print("✅ PyTorch 動作確認:", y)

# GPU の使用確認
print("🔹 PyTorch GPU 利用可能:", torch.cuda.is_available())