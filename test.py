import sys
import torch
import load_data
from model import ColorizationModel
from utils import  save_lab


def main(model_path):
    model = ColorizationModel
    model.load_state_dict(torch.load(model_path))
    model.etest()

    test_loader = load_data.get_val_data_loader('./data/test')
    for i, (img_l, img_ab, size) in enumerate(test_loader):
        if torch.cuda.is_available():
            img_l = img_l.cuda()
            img_ab = img_ab.cuda()

        output_ab = model(img_l)

        for j in range(min(len(output_ab), 5)):
            img_origin = 'img-{}-origin.jpg'.format(i * test_loader.batch_size + j)
            img_origin_gray = 'img-{}-gray.jpg'.format(i * test_loader.batch_size + j)
            save_lab(img_l[j], img_ab[j], img_origin, size[0], size[1], save_path='./outputs/test', save_gray=img_origin_gray)

            img_train = 'img-{}-train.jpg'.format(i * test_loader.batch_size + j)
            save_lab(img_l[j], output_ab[j], img_train, size[0], size[1], save_path='./outputs/test')


if __name__ == '__main__':
    main(sys.argv[1])
