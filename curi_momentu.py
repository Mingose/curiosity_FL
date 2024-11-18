import numpy as np
import matplotlib.pyplot as plt
import torch

# 参数设置
curi = 0.3
y_score_all = 1  # 假设 y_score_all 为 1
epochs = torch.arange(0, 101, dtype=torch.float32)  # 从 0 到 100 的 epoch

# 使用对数函数使增长逐渐趋缓
rare_momentum_values = 0.1 * y_score_all * torch.log1p(curi * epochs)

# 转换为 numpy 数组以便于绘图
rare_momentum_values = rare_momentum_values.numpy()

# 绘制曲线
plt.plot(epochs.numpy(), rare_momentum_values, label="rare_momentum (curi=0.3, log1p)")
plt.xlabel("Epoch")
plt.ylabel("rare_momentum")
plt.title("rare_momentum 随 epoch 的平缓增长曲线")
plt.legend()
plt.grid()
plt.show()
