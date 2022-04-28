from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import random


"""
_default = {
    "rate" : 0.85
}

# 划分数据集
path = "images/datasetnew/JPEGImages"
file_name = [name.split(".jpg")[0] for name in os.listdir(path)]
random.shuffle(file_name)
length = len(file_name)
train_set = file_name[:int(_default["rate"]*length)]
test_set = file_name[int(_default["rate"]*length):]

with open("images/train.txt", "w") as f:
    seq = "\n"
    f.write(seq.join(train_set))

with open("images/test.txt", "w") as f:
    seq = "\n"
    f.write(seq.join(test_set))
"""

def load_unet_data(is_train):
    path = os.path.join("images", "train.txt" if is_train else "test.txt")
    with open(path) as f:
        img_names = f.readlines()
    img_names = [img.strip() for img in img_names]
    images, targets = [], []
    for img in img_names:
        images.append(Image.open(os.path.join("images", "datasetnew", "JPEGImages", img + ".jpg")))
        targets.append(Image.open(os.path.join("images", "datasetnew", "SegmentationClassPNG", img + ".png")))
    return images, targets


class UnetDataset(Dataset):
    def __init__(self, transforms=None, is_train=True):
        self.images, self.targets = load_unet_data(is_train)
        self.transforms = transforms
        
    def __getitem__(self, idx):
        if self.transforms is not None:
            if self.images[idx].size[0] == 1604:
                trans = self.transforms["l"]
            elif self.images[idx].size[0] == 1136:
                trans = self.transforms["s"]
            else:
                raise ValueError("length should be 1604 or 1136")

        seed = random.randint(0, 2147483647)
        random.seed(seed)
        image = torchvision.transforms.ToTensor()(trans(self.images[idx]))
        image = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        random.seed(seed)
        target = trans(self.targets[idx])
        target = torch.tensor(np.array(target))

        return image, target
    
    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        images = torch.stack(images,dim=0)
        targets = torch.stack(targets, dim=0).squeeze(1).type(torch.long)
        return images, targets



transform = {
    "train" : transforms.Compose([
        transforms.CenterCrop((832, 1312)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ]),
    "val" : transforms.Compose([
        transforms.CenterCrop((832, 1312))
    ])
}


if __name__ == "__main__":

    train = UnetDataset(transform["train"], is_train=True)
    validation = UnetDataset(transform["val"], is_train=False)

    train_iter = DataLoader(dataset=train, batch_size=8, shuffle=True, collate_fn=train.collate_fn)
    test_iter = DataLoader(dataset=validation, batch_size=8, shuffle=False, collate_fn=validation.collate_fn)

    criterion = torch.nn.CrossEntropyLoss()
    print(len(train), len(validation))

    for X, Y in train_iter:
        print(X.shape, Y.shape)
        break
