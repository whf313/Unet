import torch
import os
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
    "batch_size": 8,
    "transforms": {
        "train" : {
            "l" : transforms.Compose([
                transforms.Resize(416),
                transforms.CenterCrop((416, 656)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ]),
            "s" : transforms.Compose([
                transforms.Resize(492),
                transforms.CenterCrop((416, 656)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ])
        },
        "val" : {
            "l" : transforms.Compose([
                transforms.Resize(416),
                transforms.CenterCrop((416, 656))
            ]),
            "s" : transforms.Compose([
                transforms.Resize(492),
                transforms.CenterCrop((416, 656))
            ])
        }
    },
    "n_epochs" : 300,
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.0001,                 # learning rate of SGD
        #'momentum': 0.9              # momentum for SGD
    },
    "save_path" : "./model/model.pth"
}


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tb_writer = SummaryWriter()
    print(f"training on {device}")

    model = Unet(config["num_classes"], config["backbone"]).to(device)
    #model.load_state_dict(torch.load("./model/model.pth"))

    train_data = UnetDataset(config["transforms"]["train"], is_train=True)
    # train_loader = DataLoader(dataset=train_data, batch_size=config["batch_size"], shuffle=True,
    #                         collate_fn=train_data.collate_fn)
    train_loader = DataLoader(dataset=train_data, batch_size=config["batch_size"], shuffle=True)

    test_data = UnetDataset(config["transforms"]["val"], is_train=False)
    test_loader = DataLoader(dataset=test_data, batch_size=config["batch_size"], shuffle=False)
    # test_loader = DataLoader(dataset=test_data, batch_size=config["batch_size"], shuffle=False,
    #                        collate_fn=test_data.collate_fn)

    # optimizer = torch.optim.SGD(model.parameters(), **config["optim_hparas"])
    optimizer = torch.optim.Adam(model.parameters(), **config["optim_hparas"])
    scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: ((0.2) ** (epoch // 20)))

    min_mIoU = 0


    for epoch in range(config["n_epochs"]):
        # train
        print("epoch:", epoch)
        scheduler.step()
        train_loss, train_mIoU = train_one_epoch(model=model,
                                                 optimizer=optimizer,
                                                 data_loader=train_loader,
                                                 device=device)

        # validate
        val_loss, val_mIoU = evaluate(model=model,
                                      data_loader=test_loader,
                                      device=device)

        tags = ["train_loss", "train_mIoU", "val_loss", "val_mIoU", "lr"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_mIoU, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_mIoU, epoch)
        tb_writer.add_scalar(tags[4], optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        if val_mIoU > min_mIoU:
            min_mIoU = val_mIoU
            torch.save(model.state_dict(), config["save_path"])
        if os.path.exists("./model") is False:
            os.makedirs("./model")
        #torch.save(model.state_dict(), config["save_path"])



if __name__ == "__main__":
    device = "cude" if torch.cuda.is_available() else "cpu"
    train(config)