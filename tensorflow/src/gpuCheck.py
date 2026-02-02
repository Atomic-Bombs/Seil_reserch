import os
# 最新GPUでのJITエラーを回避するためのフラグ
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
# エラーが出ているMLIRの最適化を一部オフにする
os.environ['TF_MLIR_GRAPH_OPTIMIZATION_LEVEL'] = '0'
import tensorflow as tf

# TensorFlow の動作確認
a = tf.constant([1.0, 2.0, 3.0])
b = tf.square(a)
print("✅ TensorFlow 動作確認:", b.numpy())

# GPU の使用確認
print("🔹 TensorFlow GPU 利用可能:", len(tf.config.list_physical_devices('GPU')) > 0)