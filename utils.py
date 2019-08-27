import torch
from PIL import Image
import matplotlib
from skimage import color

class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def save_lab(img_l, img_ab, save_name, width, height, save_path='./outputs', save_gray=None):
    if torch.cuda.is_available():
        img_l = img_l.detach().cpu()
        img_ab = img_ab.detach().cpu()

    img = torch.cat((img_l, img_ab), 0).numpy()
    img = img.transpose((1, 2, 0))
    img = color.lab2rgb(img)
    img_file = '{}/{}'.format(save_path, save_name)
    matplotlib.image.imsave(img_file, img)
    resize(img_file, width, height)

    if save_gray:
        gray = color.rgb2gray(img)
        img_file = '{}/{}'.format(save_path, save_gray)
        matplotlib.image.imsave(img_file, gray, cmap='gray')
        resize(img_file, width, height)


def resize(img_file, width, height):
    img = Image.open(img_file)
    out = img.resize((width, height), Image.ANTIALIAS)
    out.save(img_file)
