import torch
import numpy as np
import const
import skimage.color as color
import utils
from torchvision import datasets, transforms

class GraysacleImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # ignore target
        path, _ = self.imgs[index]
        img = self.loader(path)
        # rgb space
        img_rgb = self.transform(img)

        # lab space
        img_lab = color.rgb2lab(img_rgb)
        # img_lab = (img_lab + 128) / 255

        # add channle to fit image format [batch, channel, height, width]
        img_l = img_lab[:,:,0]
        img_l = torch.from_numpy(img_l).unsqueeze(0).float()

        img_ab = img_lab[:, :, 1:3]
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()

        return img_l, img_ab, img.size

def get_train_data_loader():
    transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),])
    folder = GraysacleImageFolder('./data/train', transform)
    loader = torch.utils.data.DataLoader(folder, batch_size=const.BATCH_SIZE, shuffle=True)
    return loader

def get_val_data_loader(data_path='./data/val'):
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    folder = GraysacleImageFolder(data_path, transform)
    loader = torch.utils.data.DataLoader(folder, batch_size=const.BATCH_SIZE, shuffle=False)
    return loader


if __name__ == '__main__':
    train_loader = get_val_data_loader('./data/test')
    import matplotlib
    for img_l, img_ab, size in train_loader:
        utils.save_lab(img_l[0], img_ab[0], 'test.png', size[0], size[1], save_gray='gray.png')
