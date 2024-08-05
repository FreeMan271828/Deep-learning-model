from d2l import torch as d2l, torch
import torch
from torch import nn

# 超参数
batch_size = 256

# 使用mnist数据集获取数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)
)


# 初始化权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

# 损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 优化方法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练
num_epochs = 5
d2l.train_ch3(net=net,
              train_iter=train_iter,
              test_iter=test_iter,
              loss=loss,
              num_epochs=num_epochs,
              updater=trainer)
