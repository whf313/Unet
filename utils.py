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


class Dice_Lossfn(torch.nn.Module):
    def __init__(self):
        super(Dice_Lossfn, self).__init__()

    def forward(self, pred, target, smooth = 1e-5):
        n, c, h, w = pred.size()
        target[target != 0] = 1

        temp_inputs = torch.softmax(pred.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)    # shape=(n, h*w, c)
        temp_target = target.view(n, -1)            # shape=(n, h*w)

        tp = torch.sum(temp_target * temp_inputs[..., -1], axis=[0, 1])
        fp = torch.sum(temp_inputs[..., -1], axis=[0, 1])
        fn = torch.sum(temp_target, axis=[0, 1])

        score = (2 * tp + smooth) / (fn + fp + smooth)
        d_loss = 1 - torch.mean(score)

        return d_loss


def train_one_epoch(model, optimizer, data_loader, device, lr_ratio = 0.5):
    accumulator = Accumulator()
    model.train()
    ce_Loss = torch.nn.CrossEntropyLoss()
    dice_Loss = Dice_Lossfn()
    acc_loss = torch.zeros(1).to(device)

    acc_CE = torch.zeros(1).to(device)
    acc_Dice = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        pred = model(x)
        accumulator.add(cal_IoU(pred, y))

        # loss = CE_Loss(pred, Y.type(torch.long)) + Dice_Loss(pred, Y)
        ce = ce_Loss(pred, y.type(torch.long))
        dice = dice_Loss(pred, y)
        loss = lr_ratio * ce + (1 - lr_ratio) * dice

        loss.backward()
        acc_loss += loss.detach()
        acc_CE += ce.detach()
        acc_Dice += dice.detach()

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()

    return acc_loss.item() / (step + 1), acc_CE.item() / (step + 1), acc_Dice.item() / (step + 1), accumulator.accuracy()


def evaluate(model, data_loader, device, lr_ratio = 0.5):
    model.eval()
    CE_Loss = torch.nn.CrossEntropyLoss()
    Dice_Loss = Dice_Lossfn()
    accumulator = Accumulator()
    acc_loss = torch.zeros(1).to(device)

    acc_CE = torch.zeros(1).to(device)
    acc_Dice = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
        accumulator.add(cal_IoU(pred, y))

        # loss = CE_Loss(pred, Y.type(torch.long)) + Dice_Loss(pred, Y)
        ce = CE_Loss(pred, y.type(torch.long))
        dice = Dice_Loss(pred, y)
        loss = lr_ratio * ce + (1 - lr_ratio) * dice

        acc_loss += loss.detach()
        acc_CE += ce.detach()
        acc_Dice += dice.detach()

    return acc_loss.item() / (step + 1), acc_CE.item() / (step + 1), acc_Dice.item() / (step + 1), accumulator.accuracy()


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
                      [1, 1, 1, 1, 1],
                      [0, 1, 1, 1, 0],
                      [0, 0, 1, 0, 0]]], dtype=torch.float32, requires_grad=True).unsqueeze(0)

    conv = torch.nn.Conv2d(2, 2, kernel_size=3, padding=1)


    y = torch.tensor([[0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]], dtype=torch.float32, requires_grad=True).unsqueeze(0)

    x = torch.randn([8, 2, 5, 5])
    x[x<0] = 0
    y = torch.randn([8, 5, 5])
    y[y < 0] = 0
    print(x.shape, y.shape)
    print(cal_IoU(x, y))
    dice_loss = Dice_Lossfn()
    #dice_loss = torch.nn.CrossEntropyLoss()
    temp = conv(x)
    print(conv.weight.grad)
    loss = dice_loss(temp, y.type(torch.long))
    loss.backward()
    print(loss)
    print(conv.weight.grad)

    y = torch.tensor([[0, 0, 0, 0, 0],
                      [0, 0, 0.2, 0, 0],
                      [0.7, 0, 0.5, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0.3]])

    y[y!=0] = 1


