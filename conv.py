import torch
import torch.nn as nn
import numpy as np


class Test_Conv(nn.Module):

    kernel_filter = None
    def __init__(self):
        super(Test_Conv, self).__init__()
        # self.conv = nn.Conv2d(1, 1, 3)
        ksize = 4
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=4,
            bias=False)
        self.kernel_filter = self.conv.weight.data.numpy().reshape(ksize,ksize)

    def forward(self, x):
        print("Before")
        print("size : \t",x.size())
        print("data : \n",x.to('cpu').detach().numpy().copy())
        print("\n")

        print("Calc Self Conv")
        x_np = x.to('cpu').detach().numpy().copy().reshape(4,4)
        calc_conv = 0 ;
        for col in range(self.kernel_filter.shape[0]):
            for row in range(self.kernel_filter.shape[1]):
                calc_conv += self.kernel_filter[row][col] * x_np[row][col]
        print("kernel filter :")
        print(self.kernel_filter )
        print("data : \n",calc_conv)
        print("\n")

        x = self.conv(x)
        print("After")
        print("size : \t",x.size())
        print("data : \n",x.to('cpu').detach().numpy().copy())
        print("\n")
        return x

net = Test_Conv()

#　入力
nparr = np.array([1,2,3,4]).astype(np.float32).reshape(2,2)
nparr = np.block([[nparr,nparr],[nparr,nparr]]).reshape(1,1,4,4)
input = torch.from_numpy(nparr).clone()

#　出力
out = net(input)
exit()