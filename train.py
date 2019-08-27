import os
import time
import torch
import const
from torch import nn
from utils import AverageMeter, save_lab
from model import ColorizationModel
from load_data import get_train_data_loader, get_val_data_loader

IS_GPU_AVALIABLE = torch.cuda.is_available()

def train(train_loader, model, criterion, optiomizer, epoch):
    print('starting training epoch {0} ...'.format(epoch))
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
    start = time.time()
    for i, (img_l, img_ab, size) in enumerate(train_loader):
        if IS_GPU_AVALIABLE:
            img_l = img_l.cuda()
            img_ab = img_ab.cuda()
        data_time.update(time.time() - start)

        # train model
        output_ab = model(img_l)
        loss = criterion(output_ab, img_ab)
        losses.update(loss.item(), img_l.size(0))

        # compute gradient
        optiomizer.zero_grad()
        loss.backward()
        optiomizer.step()

        # record time of forward and backward
        batch_time.update(time.time() - start)
        start = time.time()

        if i % const.LOG_CYCLE == 0:
             print('Training Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
              epoch, i, len(train_loader), batch_time=batch_time,
             data_time=data_time, loss=losses))


def validate(val_loader, model, criterion, epoch):
    model.eval()

    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
    start_time = time.time()
    save_sample_image = True
    for i, (img_l, img_ab, size) in enumerate(val_loader):
        if IS_GPU_AVALIABLE:
            img_l = img_l.cuda()
            img_ab = img_ab.cuda()
        data_time.update(time.time() - start_time)

        # train model
        output_ab = model(img_l)
        loss = criterion(output_ab, img_ab)
        losses.update(loss.item(), img_l.size(0))

        if save_sample_image:
            save_sample_image = False
            for j in range(min(len(output_ab), 5)):
                img_origin = 'img-{}-epoch-{}-origin.jpg'.format(i * val_loader.batch_size + j, epoch)
                img_origin_gray = 'img-{}-epoch-{}-gray.jpg'.format(i * val_loader.batch_size + j, epoch)
                save_lab(img_l[j], img_ab[j], img_origin, size[0], size[1], save_gray=img_origin_gray)

                img_train = 'img-{}-epoch-{}-train.jpg'.format(i * val_loader.batch_size + j, epoch)
                save_lab(img_l[j], output_ab[j], img_train, size[0], size[1])

        # record time of forward and backward
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        if i % const.LOG_CYCLE == 0:
             print('Validating Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
              epoch, i, len(val_loader), batch_time=batch_time,
             data_time=data_time, loss=losses))

    return losses.avg

def run(train_loader, val_loader, model, criterion, optiomizer):

    os.makedirs('outputs/', exist_ok=True)
    os.makedirs('outputs/test', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    if IS_GPU_AVALIABLE:
        model = model.cuda()
        criterion = criterion.cuda()

    best_losses = 1e10
    for epoch in range(const.EPOCHS):
        train(train_loader, model, criterion, optiomizer, epoch)
        with torch.no_grad():
            losses = validate(val_loader, model, criterion, epoch)

        if losses < best_losses:
            best_losses = losses
            checkpoints = 'checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch+1,losses)
            torch.save(model.state_dict(), checkpoints)


def main():
    model = ColorizationModel
    criterion = nn.MSELoss()
    optiomizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)
    train_loader = get_train_data_loader()
    val_loader = get_val_data_loader()

    run(train_loader, val_loader, model, criterion, optiomizer)


if __name__ == '__main__':
    main()
