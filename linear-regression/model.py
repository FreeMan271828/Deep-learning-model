import torch


# 基本模型
def linreg(x, w, b):
    return torch.matmul(x, w) + b


# 均方损失
def squared_loss(test_y, true_y):
    return (test_y - true_y.reshape(test_y.shape)) ** 2 / 2


# 小批量随机梯度下降(优化方法)
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            # 均方损失的梯度
            param -= lr * param.grad / batch_size
            param.grad.zero_()