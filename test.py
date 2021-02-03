from __future__ import print_function
import torch

# 計算地点の準備
z = torch.tensor([1.0, 2.0], requires_grad=True)

# f(z)の準備
f = z[0]*2 + z[1]**2

# 微分の実行
f.backward()

print(z.grad)