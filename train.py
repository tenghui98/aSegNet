import os
import numpy as np
from tqdm import tqdm
import torch

from pytorchtools import EarlyStopping
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator
from model.deepfeg2 import DeepLabv3_plus, get_1x_lr_params, get_10x_lr_params
from dataloaders import make_data_loader
from mypath import Path
from dataset_dict import dataset


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.eraly_stopping = EarlyStopping(args, patience=10, verbose=True)
        kwargs = {}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        model = DeepLabv3_plus(nInputChannels=3, n_classes=2, os=16, pretrained=True)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
        #                              weight_decay=args.weight_decay)
        train_params = [{'params': filter(lambda p: p.requires_grad, get_1x_lr_params(model)), 'lr': args.lr},
                        {'params': filter(lambda p: p.requires_grad, get_10x_lr_params(model)), 'lr': args.lr * 10}]
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=False)

        if args.use_balanced_weights:
            # classes_weights_path = os.path.join(Path.root_dir('img'), args.category, args.scene,
            #                                     args.scene + '_classes_weights.npy')
            # if os.path.isfile(classes_weights_path):
            #     weight = np.load(classes_weights_path)
            # else:
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

        self.best_pred = float('inf')
        self.flag = True
        self.early = 0
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        s1_loss = 0.0
        s4_loss = 0.0
        self.model.train()
        num_img_tr = len(self.train_loader)
        tbar = tqdm(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            preds = self.model(image)
            loss, s1_loss, s4_loss = self.criterion(preds, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            s1_loss += s1_loss.item()
            s4_loss += s4_loss.item()
            tbar.set_description('Train loss: %.5f' % (train_loss / (i + 1)))

        self.writer.add_scalars('train', {'train_loss': train_loss / num_img_tr}, epoch)
        self.writer.add_scalars('train/', {'s1_l2_loss': s1_loss / num_img_tr,
                                           's4_ce_loss': s4_loss / num_img_tr}, epoch)
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
                preds = self.model(image)
            loss, s1_loss, s4_loss = self.criterion(preds, target)
            val_loss += loss.item()
            tbar.set_description('Val loss: %.5f' % (val_loss / (i + 1)))

            if i == 0:
                self.summary.visualize_image(self.args, self.writer, image, target, preds, epoch)
            pred = preds['s1']
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
        self.writer.add_scalars('val/Stats', {'Recall': Recall,
                                              'Precision': Precision,
                                              'Fmearue': Fmeasure}, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.5f' % val_loss)
        new_pred = val_loss / num_img_val

        self.eraly_stopping(self.saver, new_pred, trainer.model)
        if self.eraly_stopping.early_stop:
            print("Early stopping")
            self.early = 1

        # if new_pred < self.best_pred:
        #     is_best = True
        #     self.flag = False
        #     filename = self.args.scene + '.pth.tar'
        #     self.best_pred = new_pred
        #     self.saver.save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': self.model.state_dict(),
        #         'optimizer': self.optimizer.state_dict(),
        #         'best_pred': self.best_pred,
        #     }, is_best, filename)

        return self.early


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
                    flag = trainer.validation(epoch)
                    if flag == 1:
                        break
            trainer.writer.close()
