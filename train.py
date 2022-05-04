import torch
import os
import time
import json
from torch.utils.data import DataLoader
from torchvision import transforms
from unet import Unet
from data import UnetDataset
from utils import train_one_epoch, evaluate
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

config = {
    "num_classes": 2,
    "backbone": "vgg16",
    "batch_size": 16,
    "transforms": {
        "train" : {
            "l" : transforms.Compose([
                transforms.Resize(208),
                transforms.CenterCrop((208, 320)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ]),
            "s" : transforms.Compose([
                transforms.Resize(246),
                transforms.CenterCrop((208, 320)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ])
        },
        "val" : {
            "l" : transforms.Compose([
                transforms.Resize(208),
                transforms.CenterCrop((208, 320))
            ]),
            "s" : transforms.Compose([
                transforms.Resize(246),
                transforms.CenterCrop((208, 320))
            ])
        }
    },
    "n_epochs" : 300,
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 5e-3,                 # learning rate of SGD
        'momentum': 0.9,              # momentum for SGD
        "weight_decay" : 1e-4
    },
    "warmup" : 20,
    "save_path" : "./model"
}

def lr_lambda(epoch):
    # warmup
    if epoch < config["warmup"]:
        return epoch / config["warmup"]
    else:
        return (0.2) ** ((epoch - config["warmup"]) // 40)

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tm = time.localtime(time.time())
    save_name = str(tm.tm_mon) + "-" + str(tm.tm_mday) + "_" + str(tm.tm_hour) + ":" + str(tm.tm_min) + "_" + "SGD" + ".pth"

    tb_writer = SummaryWriter()
    print(f"training on {device}")

    model = Unet(backbone=config["backbone"], num_class=config["num_classes"]).to(device)
    # model.load_state_dict(torch.load("./model/model.pth"))

    # train_data = UnetDataset(transforms=config["transforms"]["train"], is_train=True)
    # train_loader = DataLoader(dataset=train_data, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    #
    # test_data = UnetDataset(transforms=config["transforms"]["val"], is_train=False)
    # test_loader = DataLoader(dataset=test_data, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    optimizer = torch.optim.SGD(model.parameters(), **config["optim_hparas"])
    # optimizer = torch.optim.Adam(model.parameters(), **config["optim_hparas"])
    scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

    min_mIoU = 0

    for epoch in range(config["n_epochs"]):
        # train
        print("epoch:", epoch)
        scheduler.step()

        train_data = UnetDataset(transforms=config["transforms"]["train"], is_train=True)
        train_loader = DataLoader(dataset=train_data, batch_size=config["batch_size"], shuffle=True, num_workers=8)
        test_data = UnetDataset(transforms=config["transforms"]["val"], is_train=False)
        test_loader = DataLoader(dataset=test_data, batch_size=config["batch_size"], shuffle=False, num_workers=8)

        train_loss, ce_loss, dice_loss, train_mIoU = train_one_epoch(model=model,
                                                                     optimizer=optimizer,
                                                                     data_loader=train_loader,
                                                                     device=device,
                                                                     lr_ratio=0.5)

        # validate
        val_loss, ce_loss, dice_loss, val_mIoU = evaluate(model=model,
                                                          data_loader=test_loader,
                                                          device=device,
                                                          lr_ratio=0.5)

        tags = ["train_loss", "CE_Loss", "Dice_Loss", "train_mIoU", "val_loss", "val_mIoU", "lr"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], ce_loss, epoch)
        tb_writer.add_scalar(tags[2], dice_loss, epoch)
        tb_writer.add_scalar(tags[3], train_mIoU, epoch)
        tb_writer.add_scalar(tags[4], val_loss, epoch)
        tb_writer.add_scalar(tags[5], val_mIoU, epoch)
        tb_writer.add_scalar(tags[6], optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        if val_mIoU > min_mIoU:
            min_mIoU = val_mIoU
            torch.save(model.state_dict(), os.path.join(config["save_path"], save_name))
        if os.path.exists("./model") is False:
            os.makedirs("./model")
        # torch.save(model.state_dict(), config["save_path"])


if __name__ == "__main__":
    device = "cude" if torch.cuda.is_available() else "cpu"
    train(config)