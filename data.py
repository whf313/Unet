from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
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

        means = [0.182, 0.183, 0.183]
        stds = [0.182, 0.182, 0.181]

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
        # image = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)     # 使用ImageNet的means和std
        image = torchvision.transforms.Normalize(mean=means, std=stds)(image)                               # 使用训练集的means和std
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


config = {
    "transforms": {
        "train" : {
            "l" : transforms.Compose([
                transforms.Resize(208),
                transforms.CenterCrop((208, 320)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(contrast=(2, 2))
            ]),
            "s" : transforms.Compose([
                transforms.Resize(246),
                transforms.CenterCrop((208, 320)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(contrast=(2, 2))
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
    }
}

def get_mean_std(train_iter, test_loader):
    num_images = len(train_iter) * train_iter.batch_size + len(test_loader) * test_loader.batch_size
    means, std = [0, 0, 0], [0, 0, 0]
    for images, _ in tqdm(train_iter):
        for img in images:
            for i in range(3):
                means[i] += img[i, :, :].mean().item()
                std[i] += img[i, :, :].std().item()

    for images, _ in tqdm(test_loader):
        for img in images:
            for i in range(3):
                means[i] += img[i, :, :].mean().item()
                std[i] += img[i, :, :].std().item()


    means = [item / num_images for item in means]
    std = [item / num_images for item in std]
    print("means:", means)
    print("std:", std)


if __name__ == "__main__":

    # train = UnetDataset(transform["train"], is_train=True)
    # validation = UnetDataset(transform["val"], is_train=False)
    #
    # train_iter = DataLoader(dataset=train, batch_size=8, shuffle=True, collate_fn=train.collate_fn)
    # test_iter = DataLoader(dataset=validation, batch_size=8, shuffle=False, collate_fn=validation.collate_fn)
    #
    # criterion = torch.nn.CrossEntropyLoss()
    # print(len(train), len(validation))
    #
    # for X, Y in train_iter:
    #     print(X.shape, Y.shape)
    #     break
    train_data = UnetDataset(transforms=config["transforms"]["val"], is_train=True)
    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
    test_data = UnetDataset(transforms=config["transforms"]["val"], is_train=False)
    test_loader = DataLoader(dataset=test_data, batch_size=8, shuffle=True)
    get_mean_std(test_loader, test_loader)
    get_mean_std(test_loader, test_loader)

    means = [0.000250, 0.000250, 0.000250]
    std = [0.000188, 0.000188, 0.000188]

    # train单独的
    # means: [0.18236509055490246, 0.18299943647938421, 0.1830413048289931]
    # std: [0.18191574280186054, 0.18171371900154135, 0.18121724772768524]

    # train和test的
    # means: [0.18268681916112944, 0.18331488647036315, 0.18335400004227592]
    # std: [0.18181476445217443, 0.1816139248222476, 0.18112034400607027]

    # test单独的
    # means: [0.18444785995310858, 0.18504156010519518, 0.1850655948939292]
    # std: [0.1812620408538925, 0.18106768299874507, 0.18058992363512516]

    # colorjitter后
    # [0.25491275736375857, 0.2556106920697187, 0.2554443264870267]
    # std: [0.2845973561664945, 0.28434298501202937, 0.28348791736521217]