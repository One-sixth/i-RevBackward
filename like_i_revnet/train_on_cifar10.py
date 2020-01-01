'''
使用cifar10数据集快速测试你写的新框架
'''

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchvision.datasets.cifar import CIFAR10
try:
    from rev_net import Like_IRevNet as NewNet
except:
    from .rev_net import Like_IRevNet as NewNet


start_time = time.time()


def train(model, device, optimizer, x_train, y_train, epoch, batch_size=100):
    ids = np.arange(len(x_train))
    np.random.shuffle(ids)
    x_train = x_train[ids]
    y_train = y_train[ids]
    model.train()
    batch_count = int(np.round(len(x_train) / batch_size))
    last_show_time = time.time()
    for batch_idx in range(batch_count):
        data = np.transpose(x_train[batch_idx * batch_size: (batch_idx + 1) * batch_size], [0, 3, 1, 2])
        target = y_train[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        data = torch.tensor(data, dtype=torch.float32) / 255.
        data = data + torch.zeros(1, device=data.device, dtype=data.dtype, requires_grad=True)
        target = torch.tensor(target).long()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Time: {:.3f}'.format(
            epoch, batch_idx * len(data), len(x_train), 100. * batch_idx / batch_count, loss.item(), time.time()-last_show_time))
        last_show_time = time.time()


def test(model, device, x_val, y_val, batch_size=100):
    model.eval()
    test_loss = 0
    correct = 0
    batch_count = int(np.round(len(x_val) / batch_size))
    start_time = time.time()
    with torch.no_grad():
        for batch_idx in range(batch_count):
            data = np.transpose(x_val[batch_idx * batch_size: (batch_idx + 1) * batch_size], [0, 3, 1, 2])
            target = y_val[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            data = torch.tensor(data, dtype=torch.float32) / 255.
            target = torch.tensor(target).long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss_op
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(x_val)

    print('\nTest set: Average loss_op: {:.4f}, Accuracy: {}/{} ({:.0f}%) Test Time: {:.3f}\n'.format(
        test_loss, correct, len(x_val), 100. * correct / len(x_val), time.time()-start_time))


def main(use_rev_backward=True):

    # use_rev_backward = False

    # 定义数据集
    use_train_data_for_test = False

    dataset_path = r'D:\DeepLearningProject\datasets\cifar10'
    train_dataset = CIFAR10(dataset_path, True)
    x_train, y_train = train_dataset.data, train_dataset.targets
    val_dataset = CIFAR10(dataset_path, False)
    x_val, y_val = val_dataset.data, val_dataset.targets

    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    x_train = x_train[:]
    y_train = y_train[:]

    del train_dataset, val_dataset

    if use_train_data_for_test:
        x_val, y_val = x_train, y_train

    # Training settings

    use_cuda = torch.cuda.is_available()

    # torch.manual_seed(int(time.time()))

    device = torch.device("cuda" if use_cuda else "cpu")

    model = NewNet(use_rev_backward).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    try:
        model.load_state_dict(torch.load("cifar10_cnn.pt"))
    except (FileNotFoundError, RuntimeError):
        print('Not found save model')

    batch_size = 200

    for epoch in range(1000):
        train(model, device, optimizer, x_train, y_train, epoch, batch_size)
        test(model, device, x_val, y_val, batch_size)
        torch.save(model.state_dict(), "cifar10_cnn.pt")


if __name__ == '__main__':
    main()
