# ベースイメージとして NVIDIA の CUDA を使用
FROM nvcr.io/nvidia/tensorflow:25.01-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive

# OpenCVやMatplotlibに必要なシステムライブラリをインストール
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

    # TensorFlowのJITコンパイル問題を回避するための設定を環境変数で固定
ENV TF_CUDA_COMPUTE_CAPABILITIES=12.0
ENV CUDA_MODULE_LOADING=LAZY

# 追加のライブラリのみをインストール
RUN pip install --no-cache-dir \
    "numpy<2" \
    pandas \
    scikit-learn \
    matplotlib \
    Pillow \
    opencv-python

# 環境確認スクリプト（TensorFlow + pandas + scikit-learn）
RUN echo "/opt/venv/bin/python -c 'import tensorflow as tf; import pandas as pd; import sklearn; print(tf.__version__, tf.config.list_physical_devices('GPU'), pd.__version__, sklearn.__version__)'" > /check_tensorflow.sh
RUN chmod +x /check_tensorflow.sh

# コンテナのデフォルトコマンド
CMD ["/bin/bash"]
