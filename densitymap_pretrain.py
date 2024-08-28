import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import json
import warnings
warnings.filterwarnings("ignore")

import math
import time
import random
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets

from models import get_backbone_class
from util.semisup_dataset import ImageFolderSemiSup
from util.knn_evaluation import KNNValidation
from util.misc import *
from data_list import ImageList
from tsnecuda import TSNE
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import style

style.use('seaborn')
# plt.rcParams.update({'font.size': 70})

DATASET_CONFIG = {'cars': 196, 'flowers': 102, 'pets': 37, 'aircraft': 100, 'cub': 200, 'dogs': 120, 'mit67': 67,
                  'stanford40': 40, 'dtd': 47, 'celeba': 307, 'food11': 11, 'imagenet': 1000}


def parse_args():
    parser = argparse.ArgumentParser('argument for supervised training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')

    # dataset
    parser.add_argument('--dataset', type=str, default='cars')
    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)

    # model & method
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained model')
    parser.add_argument('--from_sl_official', action='store_true',
                        help='load from supervised imagenet-pretrained model (official PyTorch)')
    parser.add_argument('--from_ssl_official', action='store_true',
                        help='load from self-supervised imagenet-pretrained model (official PyTorch or top-conference papers)')
    
    # evaluation metric
    parser.add_argument('--e2e', action='store_true',
                        help='end-to-end finetuning')
    parser.add_argument('--knn', action='store_true',
                        help='k-NN evaluation (refer to Table 7a)')
    parser.add_argument('--topk', nargs='+', type=int, 
                        help='top-k value for k-NN evaluation')
    parser.add_argument('--label_ratio', type=float, default=1.0, 
                        help='ratio for the number of labeled sample (refer to Table 7b)')
    parser.add_argument('--multi_attribute', type=str, default='', 
                        help='multi-attribute setting for cars, aircraft, celeba dataset (refer to Table 7d)')
    
    parser.add_argument('--name', type=str, required=True, help='downstream dataset name')
    parser.add_argument('--downstream', type=str, required=True, help='sampled downstream path txt location')
    parser.add_argument('--openset', type=str, required=True, help='sampled openset path txt location')

    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.wd_scheduler = False

    # for semi-supervised results
    args.semi = False
    if args.label_ratio != 1:
        print('For a semi-supervised training, follow SimCLR and BYOL protocols that finetune whole network')
        args.semi = True
        args.e2e = True
        # args.weight_decay = 0.0
        
    args.model_name = '{}_{}'.format(args.dataset, args.model)
    if args.e2e:
        args.model_name += '_e2e'
    else:
        # linear_evaluation
        args.model_name += '_le'

    if args.from_sl_official:
        assert 'resnet' in args.model or 'efficientnet' in args.model or 'timm' in args.model
        args.model_name += '_from_sl_official'
    elif args.from_ssl_official:
        args.model_name += '_from_ssl_official'
    else:
        if not args.pretrained:
            assert args.pretrained_ckpt is None
            args.model_name += '_from_scratch'
        else:
            if args.method:
                args.model_name += '_from_{}'.format(args.method)
            else:
                raise ValueError('Specify the pretrained method')

    if args.tag:
        args.model_name += '_{}'.format(args.tag)

    args.save_folder = os.path.join(args.save_dir, args.model_name)
    # if not os.path.isdir(args.save_folder):
    #     os.makedirs(args.save_folder)

    if args.warm:
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

    if args.dataset in DATASET_CONFIG:
        args.n_cls = DATASET_CONFIG[args.dataset]
    elif args.dataset.startswith('imagenet_sub'):
        args.n_cls = 100 # dummy -> not important
    else:
        raise NotImplementedError

    # for multi_attribute experiments
    if args.dataset == 'aircraft':
        if args.multi_attribute == 'family': args.n_cls = 70
        if args.multi_attribute == 'manufacturer': args.n_cls = 30
    if args.dataset == 'cars':
        if args.multi_attribute == 'type': args.n_cls = 9
        if args.multi_attribute == 'brand': args.n_cls = 49
    if args.dataset == 'celeba':
        if args.multi_attribute in ['oval', 'smiling', 'pointy', 'young']: args.n_cls = 2

    return args


def set_loader(args):
    # construct data loader
    if args.dataset in DATASET_CONFIG:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise NotImplementedError
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.img_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(args.img_size),
                                        transforms.ToTensor(),
                                        normalize])

    if args.dataset in DATASET_CONFIG:
        if args.dataset == 'imagenet':
            traindir = os.path.join(args.data_folder, 'train') # under ~~/Data/CLS-LOC
            valdir = os.path.join(args.data_folder, 'val')
        else: # for fine-grained dataset
            if args.dataset == 'aircraft' or args.dataset == 'cars':
                traindir = os.path.join(args.data_folder, args.dataset, args.multi_attribute, 'train')
                valdir = os.path.join(args.data_folder, args.dataset, args.multi_attribute, 'test')
            elif args.dataset == 'celeba':
                traindir = os.path.join(args.data_folder, 'celeba', args.multi_attribute, 'train')
                valdir = os.path.join(args.data_folder, 'celeba', args.multi_attribute, 'test')
            else:
                traindir = os.path.join(args.data_folder, args.dataset, 'train')
                valdir = os.path.join(args.data_folder, args.dataset, 'test')
                
        if not args.semi:
            train_dataset = datasets.ImageFolder(root=traindir,
                                                 transform=train_transform)
        else:
            train_dataset = ImageFolderSemiSup(root=traindir,
                                               transform=train_transform,
                                               p=args.label_ratio)
        val_dataset = datasets.ImageFolder(root=valdir, transform=val_transform)
    else:
        raise NotImplementedError
    
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader


def set_model(args):    
    model = get_backbone_class(args.model)()
    feat_dim = model.final_feat_dim
    classifier = nn.Linear(feat_dim, args.n_cls)  # reset fc layer
    if args.method == 'mae':
        from models.dino_vit import trunc_normal_
        trunc_normal_(classifier.weight, std=0.01)
        classifier = torch.nn.Sequential(torch.nn.BatchNorm1d(feat_dim, affine=False, eps=1e-6), classifier)
        
        model.interpolate_pos_embed()
        model.set_mask_ratio(mask_ratio=0)
            
    criterion = nn.CrossEntropyLoss()
    model.cuda()

    if args.pretrained and args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = ckpt['model']

        # HOTFIX: always dataparallel during pretraining
        new_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "backbone." in k:
                k = k.replace("backbone.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        print('pretrained model loaded from: {}'.format(args.pretrained_ckpt))

    if args.from_sl_official:
        if 'vit' not in args.model:
            model.load_sl_official_weights()
            print('pretrained model loaded from PyTorch ImageNet-pretrained')
        else:
            model = get_backbone_class(args.model)(pretrained=True)
            print('pretrained model loaded from Timm, and note that finetune IN-1k from IN-21k')
    
    if args.from_ssl_official:
        if args.model == 'resnet50':
            assert args.method == 'simclr'
            model.load_ssl_official_weights()
            print('pretrained model loaded from SimCLR ImageNet-pretrained official checkpoint')
        elif 'timm_dino' in args.model:
            assert args.method == 'dino'
            model = get_backbone_class(args.model)(pretrained=True)
            print('pretrained model via DINO loaded from Timm, and note that finetune IN-1k from IN-21k')
        else:
            raise NotImplemented

                            
    return model


def validate(val_loader, model):
    model.eval()
    is_start = True

    with torch.no_grad():
        for idx, (images, _) in enumerate(tqdm(val_loader)):
            images = images.cuda()
            output = model(images)

            if is_start:
                all_output = output.data.cpu().float()
                is_start = False
            else:
                all_output = torch.cat((all_output, output.data.cpu().float()), 0)

    return all_output


def main():
    args = parse_args()
    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    downstream_txtfile = args.downstream
    openset_txtfile    = args.openset
    downstream = args.name

    os.makedirs('save/plots/{}'.format(downstream), exist_ok=True)
    
    model = set_model(args)

    batch_size = 32
    workers    = 8
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)

    preprocess = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(args.img_size),
                                        transforms.ToTensor(),
                                        normalize])

    with open(downstream_txtfile, 'r') as file:
        imagelists = file.readlines()
        imagelists = list(map(str.strip, imagelists))
    
    imagelists = [path.split(':')[0] for path in imagelists]
    database = ImageList(imagelists, preprocess=preprocess, ret_path=True)    
    data_loader = torch.utils.data.DataLoader(database, batch_size = batch_size, shuffle=False, num_workers=workers)

    # Assuming features_df is your DataFrame containing feature vectors
    # If you want to plot a subset of features, you can select those features here
    high_dim_data = validate(data_loader, model)
    # print(high_dim_data.shape)
    
    tsne = TSNE(n_components=2, perplexity=40, learning_rate=10.0, n_iter=2000, random_seed=42)
    tsne_embeddings = tsne.fit_transform(high_dim_data)

    tsne_embeddings /= np.linalg.norm(tsne_embeddings, axis=1)[:, np.newaxis]

    # Fit a Gaussian KDE to the normalized t-SNE embeddings
    kde = multivariate_normal(mean=[0, 0], cov=0.1)  # Adjust the covariance as needed
    density_values = kde.pdf(tsne_embeddings)

    # print(density_values.shape)

    # Plot the unit ring with KDE distribution
    plt.figure(figsize=(5, 5))
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=density_values, cmap='summer', alpha=0.01, s=300)
    plt.grid(False)
    # plt.colorbar()
    # plt.title('t-SNE Embeddings with Gaussian Kernel Density Estimation')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.show()
    plt.tight_layout()
    plt.savefig('save/plots/{}/pretrain_densityplot_downstream{}.pdf'.format(downstream, downstream), bbox_inches='tight', dpi=300)
    plt.close()

    

    with open(openset_txtfile, 'r') as file:
        imagelists = file.readlines()
        imagelists = list(map(str.strip, imagelists))
    
    imagelists = [path.split(':')[0] for path in imagelists]
    database = ImageList(imagelists, preprocess=preprocess, ret_path=True)    
    data_loader = torch.utils.data.DataLoader(database, batch_size = batch_size, shuffle=False, num_workers=workers)

    # Assuming features_df is your DataFrame containing feature vectors
    # If you want to plot a subset of features, you can select those features here
    high_dim_data = validate(data_loader, model)
    # print(high_dim_data.shape)
    
    tsne = TSNE(n_components=2, perplexity=40, learning_rate=10.0, n_iter=2000, random_seed=42)
    tsne_embeddings = tsne.fit_transform(high_dim_data)

    tsne_embeddings /= np.linalg.norm(tsne_embeddings, axis=1)[:, np.newaxis]

    # Fit a Gaussian KDE to the normalized t-SNE embeddings
    kde = multivariate_normal(mean=[0, 0], cov=0.1)  # Adjust the covariance as needed
    density_values = kde.pdf(tsne_embeddings)

    # print(density_values.shape)

    # Plot the unit ring with KDE distribution
    plt.figure(figsize=(5, 5))
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=density_values, cmap='summer', alpha=0.01, s=300)
    plt.grid(False)
    # plt.colorbar()
    # plt.title('t-SNE Embeddings with Gaussian Kernel Density Estimation')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.show()
    plt.tight_layout()
    plt.savefig('save/plots/{}/pretrain_densityplot_coreset{}.pdf'.format(downstream, downstream), bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    main()
