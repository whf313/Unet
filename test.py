import os
import torch
import numpy as np
import torchvision.transforms
from PIL import Image
from torch.optim import lr_scheduler
from unet import Unet as Unet, get_backbone

def test():
    path = "./images/datasetnew/JPEGImages"

    imgs = [Image.open(os.path.join(path, img_path)) for img_path in os.listdir(path)]
    info = [Image.open(os.path.join(path, img_path)).size for img_path in os.listdir(path)]
    w, h = list(zip(*info))
    print(w.count(1136), w.count(1604), set(w), len(w), set(h))

    imgs[0].show()
    temp = torchvision.transforms.CenterCrop((imgs[0].size[1] * 1.2, imgs[0].size[0]))(imgs[0])
    temp.show()

    img = torch.tensor(np.array(imgs[0]), dtype=torch.float32).permute(2, 0, 1)
    batch_img = torch.zeros([3, 852, 1312], dtype=img.dtype)
    batch_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
    print(img.shape)



if __name__ == '__main__':
   lambad_fn = lambda epoch: ((0.1) ** (epoch // 10))
   model = Unet(get_backbone("vgg16"), 2)
   optimizer = torch.optim.SGD(model.parameters(), lr = 0.0004)
   scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambad_fn)

   lr_list = []
   for epoch in range(30):
       print("epoch={}, lr={}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
       scheduler.step()
       lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

