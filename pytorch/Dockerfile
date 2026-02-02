# ベースイメージとして NVIDIA の CUDA を使用
FROM nvidia/cuda:13.0.0-base-ubuntu24.04

# Pythonをインストール
RUN apt update && apt install -y python3 python3-pip python3-venv

# 仮想環境の作成
RUN python3 -m venv /opt/venv

# 仮想環境を有効化し、pip を最新化
RUN /opt/venv/bin/pip install --upgrade pip

# 各種ライブラリのインストール
# PyTorchのインストール
RUN /opt/venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

#　pandasとscikit-learnのインストール
RUN /opt/venv/bin/pip install pandas scikit-learn

#　matplotlibのインストール
RUN /opt/venv/bin/pip install matplotlib Pillow opencv-python

# 環境変数設定
ENV PATH="/opt/venv/bin:$PATH"

# 環境確認スクリプト（PyTorch + pandas + scikit-learn）
RUN echo "/opt/venv/bin/python -c 'import torch; import pandas as pd; import sklearn; print(torch.__version__, torch.cuda.is_available(), pd.__version__, sklearn.__version__)'" > /check_pytorch.sh
RUN chmod +x /check_pytorch.sh

# コンテナのデフォルトコマンド
CMD ["/bin/bash"]