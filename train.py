import time
import os
import copy
import argparse
import pdb
import collections
import sys
import logging
from datetime import datetime
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
import pandas as pd
from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score

from classifier import create_network
from atlas import CsvDataset, collater
from loss import FocalLoss

MULTI_CLASS_NUM = 28
SCHEDULER_PATIENCE = 3
LEARNING_RATE = 1e-3
IMAGE_SIZE = 512
SCORE_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_WORKERS = 4

def train_val(
    fold,
    epoch,
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device):
    # train
    model.train()
    losses = []

    print('Train\t- fold {}, epoch {}...'.format(fold, epoch))

    with tqdm(total=len(train_loader)) as pbar:
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            inputs, gts = data['images'], data['annos']
            
            inputs = inputs.to(device=device)
            gts = gts.to(device=device)

            outputs = model(inputs)

            loss = criterion(outputs, gts)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()

            losses.append(loss.detach())
            pbar.update(1)

    # val
    with torch.no_grad():
        model.eval()
        accus = [[] for score in SCORE_THRESHOLDS]

        print('Val\t- fold {}, epoch {}...'.format(fold, epoch))

        with tqdm(total=len(val_loader)) as pbar:
            for i, data in enumerate(val_loader):
                inputs, gts = data['images'], data['annos']

                inputs = inputs.to(device=device)
                gts = gts.to(device=device)

                outputs = model(inputs)
                probs = torch.sigmoid(outputs)

                # calc f1 score
                scores = []
                for score in SCORE_THRESHOLDS:
                    results = torch.gt(probs, score)

                    f1 = f1_score(
                        gts.to(dtype=torch.long).cpu().numpy(),
                        results.detach().to(dtype=torch.long).cpu().numpy(),
                        average='micro'
                    )
                    scores.append(f1)

                [accus[i].append(scores[i]) for i in range(len(SCORE_THRESHOLDS))]
                pbar.update(1)

            accus = [np.mean(accu) for accu in accus]

        # update scheduler
        scheduler.step(np.max(accus))

    return np.mean(losses), accus

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script.')

    parser.add_argument('--folds', help='Automatically train K-Fold, set 1 to disable', type=int, default=4)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=4)
    parser.add_argument('--device', help='Device to train on', default='cuda:0')

    parser.add_argument('--network', help='Feature network type and depth', default='resnet-101')
    parser.add_argument('--pretrained', help='Finetune pretrained model', action='store_true')
    parser.add_argument('--dropout', help='Classifier dropout ratio', type=float, default=0.5)
    parser.add_argument('--label', help='Multi or single label', default='single')

    parser.add_argument('--dataset', help='Dataset path', default='./ATLAS')
    parser.add_argument('--data_root', help='Data root path', default='/data/ATLAS')
    parser.add_argument('--tag', help='Tag', default='ATLAS')

    flags = parser.parse_args(args)

    now = datetime.now()

    result_path = './{}_{}_{}_{}_{}_{}'.format(
        flags.tag,
        flags.network,
        flags.label,
        flags.folds,
        flags.dropout,
        now.strftime('%Y%m%d_%H%M%S')
    )

    summary_writer = SummaryWriter(result_path)

    alpha_file = os.path.join(flags.dataset, 'class_weight.csv')
    alpha_df = pd.read_csv(alpha_file)
    alpha = list(alpha_df['weight'])

    num_classes = MULTI_CLASS_NUM if flags.label == 'multi' else 1

    train_augmentations = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15, resample=2),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1), ratio=(0.9, 1.1)),
        transforms.ToTensor()
    ])

    val_augmentations = transforms.Compose([
        transforms.ToTensor()
    ])

    for fold in range(flags.folds):
        train_file = os.path.join(flags.dataset, 'train-{}.csv'.format(fold))
        val_file = os.path.join(flags.dataset, 'val-{}.csv'.format(fold))

        # data
        train_set = CsvDataset(
            csv_path=train_file,
            data_root=flags.data_root,
            num_classes=num_classes,
            phase='train',
            label='multi',
            augment=train_augmentations
        )
        val_set = CsvDataset(
            csv_path=val_file,
            data_root=flags.data_root,
            num_classes=num_classes,
            phase='val',
            label='multi',
            augment=val_augmentations
        )

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=flags.batch_size,
            shuffle=True,
            collate_fn=collater,
            num_workers=NUM_WORKERS
        )
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=flags.batch_size,
            shuffle=False,
            collate_fn=collater,
            num_workers=NUM_WORKERS
        )

        # model
        device = torch.device(flags.device)

        model = create_network(
            flags.network,
            pretrained=flags.pretrained,
            num_classes=num_classes,
            input_size=IMAGE_SIZE
        )

        model = model.to(device=device)
        model.training = True

        # criterion, optimizer and scheduler
        criterion = FocalLoss(
            num_classes=num_classes,
            alpha=alpha
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=SCHEDULER_PATIENCE,
            mode='max',
            verbose=True
        )

        for epoch in range(flags.epochs):
            loss, accuracy = train_val(
                fold,
                epoch,
                model,
                criterion,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                device
            )
            # print('loss: {}\naccu: {}'.format(loss, accuracy))

            # save history
            accus = {}
            for i, accu in enumerate(accuracy):
                accus['{}'.format(SCORE_THRESHOLDS[i])] = accu

            summary_writer.add_scalar('train/{}/loss'.format(fold), loss, epoch)
            summary_writer.add_scalars('val/{}/accuracy'.format(fold), accus, epoch)

            # save model
            model_path = os.path.join(result_path, 'fold{}_epoch{}.pth'.format(fold, epoch))
            torch.save(model.state_dict(), model_path)
            print('Save model: {}\n'.format(model_path))

if __name__ == '__main__':
    main()
