import random

from d2l import torch


# 生成数据集x-y
def generate_data(w, b, number):
    # 使用正态分布生成x数据集
    # 其中的x与w同长
    x = torch.normal(0, 1, size=(number, len(w)))
    # x.shape=(100*2) w.shape=(2)
    # 因为w的维度是1, 因此进行广播, w.shape=(2,2)
    y = torch.matmul(x, w) + b
    # 加噪声
    y += torch.normal(0, 0.01, y.shape)
    # reshape不会损失数据, 而是改变数据布局
    return x, y.reshape(-1, 1)


# 小批量数据读取
def data_iter(batch_size, x, y):
    # 构建test_x和test_y的随机索引
    numbers = len(x)
    indexes = list(range(numbers))
    random.shuffle(indexes)
    # 按照batch_size的间隔读取数据
    for i in range(0, numbers, batch_size):
        batch_indexes = torch.tensor(indexes[i:min(i + batch_size, numbers)])
        yield x[batch_indexes], y[batch_indexes]
