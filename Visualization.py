import os
import cv2
import h5py
import torch
import argparse
import numpy as np
from tqdm import tqdm
from model.FPN import FPN
from data.utils import decode_seg_map_sequence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default="/home/twsf/.cache/torch/checkpoints/panopticFPN_cityspaces.pth.tar", type=str)
    parser.add_argument('--img_dir', default="/home/twsf/data/Visdrone/test/images/", type=str)
    parser.add_argument('--res_dir', default="/home/twsf/data/Visdrone/test/roadSeg/", type=str)
    parser.add_argument('--plot', dest='plot',
                        help='wether plot test result image',
                        default=False, type=bool)
    args = parser.parse_args()
    return args


def transform(img):
    # height, width = img.shape[:2]
    # img = cv2.resize(img, (int(width*1.5), int(height*1.5)))
    img = img / 255.0
    img = img - [0.51, 0.535, 0.556]
    img = img / [0.196, 0.208, 0.246]
    img = np.array(img).astype(np.float32).transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    return img


def main():
    args = parse_args()
    if not os.path.exists(args.res_dir):
        os.mkdir(args.res_dir)

    # open operate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    blocks = [2, 4, 23, 3]
    model = FPN(blocks, 19, back_bone='resnet101').cuda()

    # Data
    img_list = os.listdir(args.img_dir)

    # Load trained model
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.checkpoint))
    print('====>loading trained model from ' + args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint)['state_dict'])

    model.eval()
    # results = []
    with torch.no_grad():
        for img_name in tqdm(img_list):
            img_path = os.path.join(args.img_dir, img_name)
            img = cv2.imread(img_path)
            input = transform(img).unsqueeze(0).cuda()

            output = model(input)
            pred = np.argmax(output.data.cpu().numpy(), axis=1)
            pred_rgb = decode_seg_map_sequence(img, pred, 'Cityscapes', args.plot)[0]
            # results.append(pred_rgb)
            pred_rgb = np.array(pred_rgb, dtype=np.uint8)
            # v = np.max(pred_rgb)
            binary_open = cv2.erode(pred_rgb, (kernel))  # 开操作
            binary_open = cv2.medianBlur(binary_open, 3)
            if args.show:
                img = cv2.imread(img_path)
                # show_image(img, seg, binary_open)
            cv2.imwrite(os.path.join(args.res_dir, img_name), binary_open)


if __name__ == "__main__":
    main()
