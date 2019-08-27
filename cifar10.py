import os
import random
import platform
import numpy
import subprocess
from PIL import Image
import pickle

# download from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
cifar10_path = './data/cifar-10-batches-py/'
train_batchs = [
    cifar10_path + 'data_batch_1',
    cifar10_path + 'data_batch_2',
    cifar10_path + 'data_batch_3',
    cifar10_path + 'data_batch_4',
    cifar10_path + 'data_batch_5',
]
test_batchs = [cifar10_path + 'test_batch']

def reader_cifar10(batchs, path='./data/train/images', file_prefix=''):
    """读取cifar10数据集的图片和label，并生成以序号命名的图片"""
    id = 0
    for file in batchs:
        data = {}
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')

        for i in range(len(data[b'data'])):  # 遍历图片像素数据
            id += 1
            # 读取单张图片数据
            img = data[b'data'][i]
            # 重建rgb彩色图片
            img = img.reshape(3, 32, 32)
            r = Image.fromarray(img[0]).convert('L')
            g = Image.fromarray(img[1]).convert('L')
            b = Image.fromarray(img[2]).convert('L')
            new_img = Image.merge('RGB', (r, g, b))

            # 保存图片(序号.png)
            save_file = path + '/' + file_prefix + str(id) + '.png'
            new_img.save(save_file)
        print('process {0} finished'.format(file))

def main():
    os.makedirs('./data/train/images', exist_ok=True)
    os.makedirs('./data/val/images', exist_ok=True)
    reader_cifar10(train_batchs, path='./data/train/images')
    reader_cifar10(test_batchs, path='./data/val/images')

if __name__ == '__main__':
    main()
