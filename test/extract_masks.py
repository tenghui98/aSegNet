from tqdm import tqdm
import torch
from model.deepfeg1 import DeepLabv3_plus
# from model.unet.unet_model import UNet
import os
import numpy as np
from myparser import parser
from mypath import  Path
from dataloaders import make_data_loader
from dataset_dict import dataset
import imageio
from PIL import Image

model_main_dir = Path.root_dir('model')
args = parser()
model = DeepLabv3_plus(nInputChannels=3, n_classes=2, os=16, pretrained=True)
# model = UNet(n_channels=3, n_classes=1)
for category,scene_list in dataset.items():
    for scene in scene_list:
        args.category = category
        args.scene = scene
        model_path = os.path.join(model_main_dir, category, scene,'model_best.pth.tar')

        ROI_path = os.path.join(Path.root_dir('img'),category,scene,'ROI.bmp')
        ROI = Image.open(ROI_path).convert('L')
        ROI = np.array(ROI).astype(np.float32)
        idx = np.where(ROI == 0.0)

        result_dir = os.path.join(Path.root_dir('result'), category, scene)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])
        model = model.cuda()
        model.eval()
        _,_,testloader,_ = make_data_loader(args)
        tbar = tqdm(testloader)

        img_idx = 0
        for i, img in enumerate(tbar):
            if args.cuda:
                img = img.cuda()
            with torch.no_grad():
                pred = model(img)
                # outputs = model(img)

            # pred = outputs['s1']
            # pred = torch.squeeze(torch.sigmoid(pred),1)
            #
            # out = (pred > args.th).cpu().numpy().astype('int')
            # out *= 255
            # out = out.astype(np.uint8)

            pred = pred.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred *= 255
            out = pred.astype(np.uint8)

            for jj in range(out.shape[0]):
                out[jj][idx] = 0.
                img_idx += 1
                fname = 'bin'+"%06d" % img_idx +'.png'
                imageio.imwrite(os.path.join(result_dir,fname),out[jj])

            tbar.set_description(args.category + '/' + args.scene)






