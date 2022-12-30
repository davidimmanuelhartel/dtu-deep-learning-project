# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import cv2
import sys
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image
import numpy as np

import torch

import util.misc as utils

from models import build_model
# from datasets.face import make_face_transforms

import matplotlib.pyplot as plt
import time


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b

def get_images(in_path):
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))

    return img_files


def get_args_parser():
    
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save the results, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--thresh', default=0.5, type=float)

    # size of the spectrograms 
    parser.add_argument('--step_size', default = 0.125, type=float)
    parser.add_argument('--nfft', default = 512, type=int)

    return parser


@torch.no_grad()
def infer(data_loader, model, postprocessors, device, output_path):
    model.eval()
    duration = 0
    results = open(os.path.join('test', 'output', 'results.txt'), 'w')
    
    # Create dictionary with segment indexes in train split
    dm_train = dm.train
    dm_train_indexes = {}
    for i, train in enumerate(dm_train):
        dm_train_indexes[train['record']] = i

    for i, (data, events, records, *_) in enumerate(data_loader):
        
        #spectogram to test
        spectogram = data.to(device)

        #image for plotting
        from sklearn.preprocessing import MinMaxScaler
        spectogram_img = np.squeeze(data).numpy()
        scaler = MinMaxScaler(feature_range=(0,1))
        spectogram_img[0] = scaler.fit_transform(spectogram_img[0].reshape(-1, spectogram_img[0].shape[-1])).reshape(spectogram_img[0].shape)
        scaler = MinMaxScaler(feature_range=(0,1))
        spectogram_img[1] = scaler.fit_transform(spectogram_img[1].reshape(-1, spectogram_img[1].shape[-1])).reshape(spectogram_img[1].shape)
        scaler = MinMaxScaler(feature_range=(0,1))
        spectogram_img[2] = scaler.fit_transform(spectogram_img[2].reshape(-1, spectogram_img[2].shape[-1])).reshape(spectogram_img[2].shape)

        spectogram_img = np.flipud(np.transpose(spectogram_img, (1, 2, 0)))

        # import torch
        # normalized_spectogram =  torch.from_numpy(spectogram_img.astype(np.float32) * 255).permute(2, 0, 1).unsqueeze(0)
        # normalized_spectogram = normalized_spectogram.to(device)
        
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        hooks = [
            model.backbone[-2].register_forward_hook(
                        lambda self, input, output: conv_features.append(output)

            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                        lambda self, input, output: enc_attn_weights.append(output[1])

            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                        lambda self, input, output: dec_attn_weights.append(output[1])

            ),

        ]
        start_t = time.perf_counter()
        outputs = model(spectogram)
        end_t = time.perf_counter()

        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()
        
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        # keep = probas.max(-1).values > 0.85
        keep = probas.max(-1).values > args.thresh

        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], (spectogram_img.shape[1], spectogram_img.shape[0]))
        
        predicted_boxes = outputs['pred_boxes'][0, keep]
        predicted_events = []

        for box in predicted_boxes:
            predicted_events.append(
                torch.tensor([box[0] - box[2]/2, box[0] + box[2]/2, 0.0])
            )
            
        predicted_events = torch.stack(predicted_events)

        fig, ax = dm_train.plot_signals(dm_train_indexes[records[0]], predicted_events=predicted_events)

        fig.savefig(os.path.join('test', 'output', f'signals_{records[0]}.png'), bbox_inches='tight', pad_inches=0, dpi = 200)

        plt.clf()
        plt.close(fig=fig)

        # real boxes
        real_bboxes = []
        for idx, t in enumerate(events):
            for el in t:
                real_bboxes.append(
                    torch.tensor([el[:2].mean(), 0.5, el[:2].diff(), 1.0])
                ) 

        real_bboxes = torch.stack(real_bboxes)
        
        real_bboxes_scaled = rescale_bboxes(real_bboxes, (spectogram_img.shape[1], spectogram_img.shape[0]))

        results.write(str(records[0]) + '\n')
        results.write('Number of events in this segment:' + str(sum([ev.shape[0] for ev in events])) + '\n')
        results.write('Unscaled real boxes:' + '\n')
        results.write(str(real_bboxes) + '\n')
        results.write('Scaled real boxes:' + '\n')
        results.write(str(real_bboxes_scaled) + '\n')
        results.write('Number of predicted boxes:' + str(bboxes_scaled.shape[0]) + '\n')
        results.write('Unscaled predicted boxes:' + '\n')
        results.write(str(outputs['pred_boxes'][0, keep]) + '\n')
        results.write('Scaled predicted boxes:' + '\n')
        results.write(str(bboxes_scaled) + '\n')

        probas = probas[keep].cpu().data.numpy()

        for hook in hooks:
            hook.remove()

        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0].cpu()

        # get the feature map shape
        h, w = conv_features['0'].tensors.shape[-2:]

        if len(bboxes_scaled) == 0:
            continue

        img = np.array(spectogram_img)*255
        img = img.astype(np.int32).copy() 

        # Plotting predicted bboxes
        for idx, box in enumerate(bboxes_scaled):
            bbox = box.cpu().data.numpy()
            bbox = bbox.astype(np.int32)
            bbox = np.array([
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
                ])
            bbox = bbox.reshape((4, 2))
            cv2.polylines(img, [bbox], True, (0, 255, 0), 2)
        
        # Plotting real bboxes
        for idx, box in enumerate(real_bboxes_scaled):
            bbox = box.cpu().data.numpy()
            bbox = bbox.astype(np.int32)
            bbox = np.array([
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
                ])
            bbox = bbox.reshape((4, 2))
            cv2.polylines(img, [bbox], True, (0, 0, 255), 2)

        img_save_path = os.path.join('test', 'output', 'stft_{}.png'.format(records[0]))
        # img = cv2.rotate(img, cv2.ROTATE_180)

        cv2.imwrite(img_save_path, img)
    
    results.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    # image_paths = get_images(args.data_path)

    from mros_data.datamodule.transforms import STFTTransform
    from mros_data.datamodule import SleepEventDataModule
    
    step = args.step_size
    nfft_param = args.nfft

    params = dict(
        data_dir="test/input/",
        batch_size=1,
        n_eval=0,
        n_test=0,
        num_workers=0,
        seed=1337,
        events={"sdb": "Sleep-disordered breathing"},
        window_duration=600,  # seconds
        cache_data=True,
        default_event_window_duration=[15],
        event_buffer_duration=3,
        factor_overlap=2,
        fs=64,
        matching_overlap=0.5,
        n_jobs=-1,
        n_records=10,
        picks=["nasal", "abdo", "thor"],
        # transform=MultitaperTransform(128, 0.5, 35.0, tw=8.0, normalize=True),
        transform=STFTTransform(
            fs=64, segment_size=int(4.0 * 64), step_size=int(step*64), nfft=nfft_param, normalize=True
        ),
        scaling="robust",
    )
    dm = SleepEventDataModule(**params)
    dm.setup('fit')

    data_loader_train= dm.train_dataloader()

    infer(data_loader_train, model, postprocessors, device, args.output_dir)
