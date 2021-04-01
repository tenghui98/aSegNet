import os
import numpy as np
from tqdm import tqdm
import torch
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator
# from model.deepfeg11 import DeepLabv3_plus
from model.unet.unet_model import UNet
from model.unet.unet_parts import dcrf
from dataloaders import make_data_loader
from mypath import Path
from dataset_dict import dataset
from model.pytorchtools import EarlyStopping


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.eraly_stopping = EarlyStopping(patience=20, verbose=True)
        kwargs = {}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        model = UNet(n_channels=3, n_classes=1, bilinear=True)
        # model = DeepLabv3_plus(nInputChannels=3, pretrained=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=False)

        if args.use_balanced_weights:
            weight = calculate_weigths_labels(args, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
            if args.cuda:
                weight = weight.cuda()
        else:
            weight = None

        self.evaluator = Evaluator(self.nclass)
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        # self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        if args.cuda:
            self.model = self.model.cuda()

        self.best_pred = 0.0
        self.flag = True
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        num_img_tr = len(self.train_loader)
        tbar = tqdm(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            pred = self.model(image)
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.5f' % (train_loss / (i + 1)))

        self.writer.add_scalars('train', {'train_loss': train_loss / num_img_tr}, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.5f' % train_loss)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        val_loss = 0.0
        num_img_val = len(self.val_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                pred = self.model(image)
            loss = self.criterion(pred, target)
            val_loss += loss.item()
            tbar.set_description('Val loss: %.5f' % (val_loss / (i + 1)))
            if i == 0:
                self.summary.visualize_image(self.args, self.writer, image, target, pred, epoch)
            pred = torch.squeeze(torch.sigmoid(pred), 1)
            pred = pred.data.cpu().numpy()
            pred = (pred > 0.7).astype('int')
            target = target.cpu().numpy()
            # pred = pred.data.cpu().numpy()
            # target = target.cpu().numpy()
            # pred = np.argmax(pred, axis=1)

            self.evaluator.add_batch(target, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        Recall, Precision, Fmeasure = self.evaluator.Stats()
        self.writer.add_scalars('train', {'val_loss': val_loss / num_img_val}, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalars('val/Stats', {'Recall': Recall,
                                              'Precision': Precision,
                                              'Fmearue': Fmeasure}, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.5f' % val_loss)

        new_pred = Fmeasure
        if new_pred > self.best_pred:
            is_best = True
            self.flag = False
            filename = self.args.scene + '.pth.tar'
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best, filename)
        if epoch == self.args.epochs - 1 and self.flag:
            is_best = False
            filename = self.args.scene + '.pth.tar'
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best, filename)
        return val_loss / num_img_val


if __name__ == '__main__':
    from myparser import parser

    args = parser()
    print(args)
    torch.manual_seed(args.seed)

    for category, scene_list in dataset.items():
        for scene in scene_list:
            args.category = category
            args.scene = scene
            trainer = Trainer(args)
            print('Starting Epoch:', trainer.args.start_epoch)
            print('Total Epoches:', trainer.args.epochs)
            for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
                trainer.training(epoch)
                if epoch % args.eval_interval == (args.eval_interval - 1):
                    val_loss = trainer.validation(epoch)
                    trainer.eraly_stopping(val_loss, trainer.model)
                    if trainer.eraly_stopping.early_stop:
                        print("Early stopping")
                        break
            trainer.writer.close()
