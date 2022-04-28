import sys
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self):
        self.right = 0.0
        self.all = 0.0

    def add(self, num):
        num_right, num_all = num
        self.right += num_right
        self.all += num_all

    def reset(self):
        self.right = 0.0
        self.all = 0.0

    def accuracy(self):
        return (self.right / self.all) * 100 if self.all else 0

    def __getitem__(self, idx):
        return (self.right[idx], self.all[idx])



def set_axes(axes, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        print(legend)
        axes.legend(legend)
    axes.grid()

def animate(X, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear'):
    assert len(set([len(x) for x in X])) == 1
    fig, axes = plt.subplots(1, 1,  tight_layout=True)
    for content, label in zip(X, legend):
        axes.plot(np.arange(1, len(content)+1), content, label=label)
    set_axes(axes, xlim, ylim, xscale, yscale, legend)
    plt.show()


def cal_IoU(pred, target):
    # type: pred:Tensor[B, channel, H, W]   target:Tensor[B, H, W]
    pred = torch.argmax(pred, dim=1)
    assert pred.shape == target.shape
    num_intersection = int(((pred * target) != 0).sum())
    num_union = int(((pred + target) != 0).sum())
    # num_union = target.numel()
    return num_intersection, num_union

def train_one_epoch(model, optimizer, data_loader, device):
    accumulator = Accumulator()
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    acc_loss = torch.zeros(1).to(device)
    #optimizer.zero_grad()

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        X, Y = data
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()

        pred = model(X)
        accumulator.add(cal_IoU(pred, Y))

        loss = criterion(pred, Y.type(torch.long))
        loss.backward()
        acc_loss += loss.detach()

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()

    return acc_loss.item() / (step + 1), accumulator.accuracy()


def evaluate(model, data_loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    accumulator = Accumulator()
    acc_loss = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
        accumulator.add(cal_IoU(pred, y))

        loss = criterion(pred, y.type(torch.long))
        acc_loss += loss.detach()

    return acc_loss.item() / (step + 1), accumulator.accuracy()


if __name__ == "__main__":
    # accumulator = Accumulator()
    # a = torch.randn([8, 3, 64, 64])
    # t = torch.zeros([8, 64, 64])
    # accumulator.add(cal_precision(a, t))
    # print(accumulator.accuracy())
    #
    # acc = Accumulator()
    # print(acc.accuracy())

    # num_epochs = 50
    #
    # train_acc = []
    # train_loss = []
    # test_acc = []
    # test_loss = []
    # for epoch in range(num_epochs):
    #     train_acc.append(random.randint(0, 99))
    #     train_loss.append(random.uniform(0, 3))
    #     test_acc.append(random.randint(0, 99))
    #     test_loss.append(random.uniform(0, 3))
    # animate((train_acc, test_acc),['train_acc', 'test_acc'], xlim=(0, len(train_acc)), ylim=(0, 100))
    # animate((train_loss, test_loss), ['train_loss', 'test_loss'],xlim=(0, len(train_loss)), ylim=(0, 3))

    x = torch.tensor([[[1, 1, 0, 1, 1],
                       [1, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 1],
                       [1, 1, 0, 1, 1]],
                      [[0, 0, 1, 0, 0],
                      [0, 1, 1, 1, 0],
                      [1, 1, 0, 1, 1],
                      [0, 1, 1, 1, 0],
                      [0, 0, 1, 0, 0]]]).unsqueeze(0)

    y = torch.tensor([[0, 0, 1, 0, 1],
                      [0, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1],
                      [0, 1, 1, 1, 0],
                      [0, 1, 1, 0, 0]]).unsqueeze(0)
    print(x.shape, y.shape)
    a,b = cal_IoU(x, y)
    print(a, b)


