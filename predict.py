from unet import Unet
import torch
from PIL import Image
from train import config
from torchvision import transforms
import os

def predict():
    path1 = "./model/model.pth"
    path2 = "./model/1/model4_lr_sche1.pth"
    img_path = "./images/datasetnew/JPEGImages"
    class_path = "./images/datasetnew/SegmentationClassPNG"
    img_name = "IMG_20210222_4_2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m1 = Unet().to(device)
    m2 = Unet().to(device)
    m1.load_state_dict(torch.load(path1, map_location=torch.device('cpu')))
    m2.load_state_dict(torch.load(path2, map_location=torch.device('cpu')))

    image = Image.open(os.path.join(img_path, img_name) + '.jpg')
    X = config["transforms"]["val"]["l"](image) if image.size[0] == 1604 else config["transforms"]["val"]["s"](image)
    Y = Image.open(os.path.join(class_path, img_name) + '.png')
    image = config["transforms"]["val"]["l"](image) if image.size[0] == 1604 else config["transforms"]["val"]["s"](image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

    p1 = m1(image.unsqueeze(0))
    p1 = torch.argmax(p1, dim=1).type(torch.float32)
    p1 = transforms.ToPILImage()(p1)

    p2 = m2(image.unsqueeze(0))
    p2 = torch.argmax(p2, dim=1).type(torch.float32)
    p2 = transforms.ToPILImage()(p2)

    Y = config["transforms"]["val"]["l"](Y) if Y.size[0] == 1604 else config["transforms"]["val"]["s"](Y)

    p1.show(), p2.show(), Y.show(), X.show()

if __name__ == '__main__':
    predict()