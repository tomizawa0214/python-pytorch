import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Test_Pooling(nn.Module):

    def __init__(self):
        super(Test_Pooling, self).__init__()
    def forward(self, x):
        print("Before")
        print("size : \t",x.size())
        print("data : \n",x.to('cpu').detach().numpy().copy())
        print("\n")

        x = F.max_pool2d(x, (2, 2))
        print("After")
        print("size : \t",x.size())
        print("data : \n",x.to('cpu').detach().numpy().copy())
        print("\n")
        return x

net = Test_Pooling()

#　入力
nparr = np.array([1,2,3,4]).astype(np.float32).reshape(2,2)
nparr = np.block([[nparr,nparr],[nparr,nparr]]).reshape(1,1,4,4)
input = torch.from_numpy(nparr).clone()

#　出力
out = net(input)