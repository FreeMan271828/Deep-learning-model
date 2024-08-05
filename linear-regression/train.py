import torch
from dateset import generate_data, data_iter
from early_stop import EarlyStopping
from model import linreg, squared_loss, sgd

# 设置网络名称以及损失函数
net = linreg
loss = squared_loss
# 真实的参数
true_w = torch.tensor([-2.4, 3])
true_b = torch.tensor(1)
# 生成训练用数据
train_w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
train_b = torch.zeros(1, requires_grad=True)
train_x, train_y = generate_data(true_w, true_b, 200)
# 训练用超参数
lr = 0.1
num_epochs = 30
batch_size = 10
early_stopping = EarlyStopping(patience=3)
# 按照轮次进行训练
for epoch in range(num_epochs):
    # 获取小批量
    for batch_x, batch_y in data_iter(batch_size, train_x, train_y):
        # 计算均方损失
        prediction = net(batch_x, train_w, train_b)
        l = loss(prediction, batch_y)
        # 计算均方损失的梯度并存储在模型参数.grad中
        l.sum().backward()
        sgd([train_w, train_b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(train_x, train_w, train_b), train_y)
        print(
            f'epoch {epoch + 1}, loss {float(train_l.mean()):f}, train_w {train_w.data}, train_b {float(train_b)}')
        early_stopping(float(train_l.mean().item()))
        if early_stopping.early_stop:
            print("Early stopping")
            break
