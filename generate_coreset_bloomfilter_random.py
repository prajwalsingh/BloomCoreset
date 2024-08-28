import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
from time import time
from tqdm import tqdm
from functools import reduce
# from fastbloom_rs import BloomFilter
# from fastbloom_rs import CountingBloomFilter
from countingbloom import CBloomFilter
import matplotlib.pyplot as plt
from matplotlib import style
import multiprocessing as mp
import sys
from random import shuffle
from scipy.stats import entropy
import math
import open_clip
import torchvision
import torch
from data_list import ImageList
import pre_process as prep
from torch.autograd import Variable
from glob import glob
from natsort import natsorted
import argparse
from kmeans_gpu import KMeans

np.random.seed(45)

style.use('ggplot')


def predict_hash_code(model, data_loader):  # data_loader is database_loader or test_loader
    model.eval()
    is_start = True
    # for i, (input, label) in enumerate(data_loader):
    for i, (input, path) in enumerate(tqdm(data_loader)):
        input = Variable(input).cuda()
        # label = Variable(label).cuda()
        with torch.no_grad():
            y = model.encode_image(input)
            # y = torch.ones(size=(512, 512))

        y /= y.norm(dim=-1, keepdim=True)

        if is_start:
            all_output = y.data.cpu().float()
            # all_label = label.float()
            all_path   = list(path)
            is_start = False
        else:
            all_output = torch.cat((all_output, y.data.cpu().float()), 0)
            # all_label = torch.cat((all_label, label.float()), 0)
            all_path.extend(list(path))

    # return all_output.cpu().numpy(), all_label.cpu().numpy()
    return all_output.cpu().numpy(), np.array(all_path)


def generate_code(model, data_loader, dataset_name, dataset_loc):
    print('Waiting for generate the hash code from database')
    data_hash, data_path = predict_hash_code(model, data_loader)
    np.save('{}/bloomssl_cache/CLIP/{}/{}_clip_hash_512.npy'.format(dataset_loc, dataset_name, dataset_name), data_hash)
    np.save('{}/bloomssl_cache/CLIP/{}/{}_clip_path_512.npy'.format(dataset_loc, dataset_name, dataset_name), data_path)
    print('generated {} hash size: '.format(dataset_name), data_hash.shape)



def parse_args():
    parser = argparse.ArgumentParser('argument for coreset sampling')

    parser.add_argument('--downstream', type=str, required=True, help='downstream dataset name')
    parser.add_argument('--openset', type=str, required=True, help='openset dataset name')
    parser.add_argument('--dataset_loc', type=str, default='../dataset', help='openset dataset name')
    parser.add_argument('--percent', type=float, default=0.01, help='openset dataset name')
    parser.add_argument('--clip_batch', type=int, default=512, help='batch size for inferencing clip encoding')
    parser.add_argument('--clip_workers', type=int, default=16, help='number of workers for inferencing clip encoding')
    parser.add_argument('--with_freq', type=bool, default=True, help='store filter data with frequency count')
    
    args = parser.parse_args()

    return args



if __name__ == "__main__":

    args = parse_args()

    downstream_data_name = args.downstream #'aircraft'
    openset_name         = args.openset #'imagenet'
    batch_size           = args.clip_batch # 512
    workers              = args.clip_workers #16
    percentage_sample    = args.percent #0.01 # 0.01, 0.02, 0.05
    dataset_loc          = args.dataset_loc

    hashtype             = 'clip'
    downstream_path_list = '{}/bloomssl_cache/Paths/{}_path_list.txt'.format(dataset_loc, downstream_data_name)
    openset_path_list    = '{}/bloomssl_cache/Paths/{}_path_list.txt'.format(dataset_loc, openset_name)

    if not os.path.isfile(downstream_path_list):
        print('Building {} path list...'.format(downstream_data_name))
        os.makedirs('{}/bloomssl_cache/Paths/'.format(dataset_loc), exist_ok=True)
        dataset_path = '{}/{}/train/**/*.*'.format(dataset_loc, downstream_data_name)
        dataset_path = natsorted(glob(dataset_path, recursive=True))

        # if len(dataset_path) == 0:
        #     dataset_path = '{}/{}/train/*'.format(dataset_loc, downstream_data_name)
        #     dataset_path = natsorted(glob(dataset_path))

        with open(downstream_path_list, 'w') as file:
            paths = list(map(lambda x: x+'\n', dataset_path))
            file.writelines(paths)
    
    if not os.path.isfile(openset_path_list):
        print('Building {} path list...'.format(openset_name))
        os.makedirs('{}/bloomssl_cache/Paths/'.format(dataset_loc), exist_ok=True)
        openset_data_dict = {'imagenet': 'imagenet1k/train_images', 'coco': 'COCO/train', 'inaturalistmini':'inaturalistmini/train', 'places365':'places365/train/'}
        if openset_name == 'all':
            for key in openset_data_dict:
                dataset_path = '{}/{}/**/*.*'.format(dataset_loc, openset_data_dict[key])
                with open(openset_path_list, 'a') as file:
                    paths = list(map(lambda x: x+'\n', natsorted(glob(dataset_path, recursive=True))))
                    file.writelines(paths)
        else:
            dataset_path = '{}/{}/**/*.*'.format(dataset_loc, openset_data_dict[openset_name])
            with open(openset_path_list, 'w') as file:
                paths = list(map(lambda x: x+'\n', natsorted(glob(dataset_path, recursive=True))))
                file.writelines(paths)

    if not os.path.isfile('{}/bloomssl_cache/CLIP/{}/{}_clip_hash_512.npy'.format(dataset_loc, downstream_data_name, downstream_data_name)):
        
        print('Building {} downstream data hash...'.format(downstream_data_name))

        os.makedirs('{}/bloomssl_cache/CLIP/{}'.format(dataset_loc, downstream_data_name), exist_ok=True)
        
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        model = model.to('cuda')

        database = ImageList(open(downstream_path_list).readlines(), preprocess=preprocess, ret_path=True)    
        data_loader = torch.utils.data.DataLoader(database, batch_size = batch_size, shuffle=False, num_workers=workers)

        generate_code(model, data_loader, downstream_data_name, dataset_loc)
        ## Reading downstream data
        print('Loading {} downstream data hash...'.format(downstream_data_name))
        downstream_code = np.load('{}/bloomssl_cache/CLIP/{}/{}_clip_hash_512.npy'.format(dataset_loc, downstream_data_name, downstream_data_name))
        downstream_path = np.load('{}/bloomssl_cache/CLIP/{}/{}_clip_path_512.npy'.format(dataset_loc, downstream_data_name, downstream_data_name), allow_pickle=False).tolist()
    else:
        ## Reading downstream data
        print('Loading {} downstream data hash...'.format(downstream_data_name))
        downstream_code = np.load('{}/bloomssl_cache/CLIP/{}/{}_clip_hash_512.npy'.format(dataset_loc, downstream_data_name, downstream_data_name))
        downstream_path = np.load('{}/bloomssl_cache/CLIP/{}/{}_clip_path_512.npy'.format(dataset_loc, downstream_data_name, downstream_data_name), allow_pickle=False).tolist()


    if not os.path.isfile('{}/bloomssl_cache/CLIP/{}/{}_clip_hash_512.npy'.format(dataset_loc, openset_name, openset_name)):
        print('Building {} openset data hash...'.format(openset_name))

        os.makedirs('{}/bloomssl_cache/CLIP/{}'.format(dataset_loc, openset_name), exist_ok=True)
        
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        model = model.to('cuda')

        database = ImageList(open(openset_path_list).readlines(), preprocess=preprocess, ret_path=True)    
        data_loader = torch.utils.data.DataLoader(database, batch_size = batch_size, shuffle=False, num_workers=workers)

        generate_code(model, data_loader, openset_name, dataset_loc)

        ## Reading openset data
        print('Loading {} openset data hash...'.format(openset_name))
        openset_code = np.load('{}/bloomssl_cache/CLIP/{}/{}_clip_hash_512.npy'.format(dataset_loc, openset_name, openset_name))
        openset_path = np.load('{}/bloomssl_cache/CLIP/{}/{}_clip_path_512.npy'.format(dataset_loc, openset_name, openset_name), allow_pickle=False).tolist()

    else:
        ## Reading openset data
        print('Loading {} openset data hash...'.format(openset_name))
        openset_code = np.load('{}/bloomssl_cache/CLIP/{}/{}_clip_hash_512.npy'.format(dataset_loc, openset_name, openset_name))
        openset_path = np.load('{}/bloomssl_cache/CLIP/{}/{}_clip_path_512.npy'.format(dataset_loc, openset_name, openset_name), allow_pickle=False).tolist()


    # import pdb; pdb.set_trace()

    downstream_size = downstream_code.shape[0]
    openset_size    = openset_code.shape[0]

    match_score = 0
    openset_code_filter = []
    openset_path_filter = []

    n = ( 10000 * (downstream_size/3500) ) #2000000  # no of items to add
    p = 0.01
    counter_size = 32
    bloom = CBloomFilter(n, counter_size=counter_size, p=p)


    print("Size of bit array:{}".format(bloom.m))
    print("Size of each bucket:{}".format(bloom.N))
    print("Number of hash functions:{}".format(bloom.k))

    threshold_mean      = 0
    code_length         = 512
    
    def select(data):
        idx = 0
        return data[idx:idx+code_length]
    
    downstream_code = np.array(list(map(select, downstream_code)))
    openset_code = np.array(list(map(select, openset_code)))

    downstream_bit_code = np.where(downstream_code < threshold_mean, 0, 1)
    openset_bit_code    = np.where(openset_code < threshold_mean, 0, 1)

    print(downstream_bit_code.shape, openset_bit_code.shape)
    print(downstream_bit_code[0])
    print(openset_bit_code[0])

    if not os.path.isdir('{}/bloomssl_cache/Filter/{}_cache'.format(dataset_loc, downstream_data_name)):
        os.makedirs('{}/bloomssl_cache/Filter/{}_cache'.format(dataset_loc, downstream_data_name))

    opensetcode_path = '{}/bloomssl_cache/Filter/{}_cache/downstream_ssl_filter_{}_{}_opensetcode_{}_codelength_{}.npy'.format(dataset_loc, downstream_data_name, downstream_data_name, openset_name, hashtype, code_length)
    opensetpath_path = '{}/bloomssl_cache/Filter/{}_cache/downstream_ssl_filter_{}_{}_opensetpath_{}_codelength_{}.npy'.format(dataset_loc, downstream_data_name, downstream_data_name, openset_name, hashtype, code_length)

    if os.path.isfile(opensetcode_path):
        print('Loading pre-computed data....')
        openset_code_filter = np.load(opensetcode_path)
        openset_path_filter = np.load(opensetpath_path)

    else:

        for hash_code in tqdm(downstream_bit_code):
            hash_code = reduce(lambda x,y: str(x)+str(y), hash_code.tolist())
            bloom.add(hash_code)

        for idx, hash_code in enumerate(tqdm(openset_bit_code)):
            hash_code = reduce(lambda x,y: str(x)+str(y), hash_code.tolist())
            # if hash_code in bloom:
            if bloom.check(hash_code):
                openset_code_filter.append(openset_code[idx])
                openset_path_filter.append(openset_path[idx])
                match_score += 1
            
        
        openset_code_filter = np.array(openset_code_filter)
        openset_path_filter = np.array(openset_path_filter)

        print('Total hit: {}'.format(match_score))
        print('Openset filter code: {}'.format(openset_code_filter.shape))

        if not os.path.isfile(opensetcode_path):
            np.save(opensetcode_path, openset_code_filter)
            np.save(opensetpath_path, openset_path_filter)


    # # Save affinity matrix plot
    # print('Calculating affinity matrix for bloom samples.....')
    # os.makedirs('data/AffinityMat_max/{}/'.format(downstream_data_name), exist_ok=True)
    # affinity_mat = downstream_code @ openset_code_filter.T
    # plt.figure(figsize=(15, 10))
    # plt.imshow(affinity_mat, cmap='viridis', aspect=str(openset_code_filter.shape[0]/downstream_code.shape[0]))
    # plt.colorbar(shrink=1.0)
    # plt.grid(False)
    # plt.title('Bloom Samples', fontsize=25)
    # plt.xlabel('Openset Bloom Data', fontsize=25)
    # plt.ylabel('Downstream Data', fontsize=25)
    # # plt.yticks(ticks=np.linspace(1, downstream_code.shape[0], 200))
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.tight_layout()
    # plt.savefig('data/AffinityMat_max/{}/downstream_{}_codelength_{}_hash_bloomsamples_{}.pdf'.format(downstream_data_name, downstream_data_name, code_length, openset_name), bbox_inches='tight', dpi=300)
    # plt.clf()
    # plt.close('all')
    # #####

    coreset_size = 0
    tau = 0.4
    filter_openset_size = math.ceil(percentage_sample*openset_size)
    n_clusters = 2048#int(1e4)
    # K = 0 # Intital K
    # N = openset_code_filter.shape[0]
    # thresh_lst = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    # freq_dict = {}

    print('Building the coreset list....')
    start_time = time()
    simscore_idx  = np.random.choice(openset_code_filter.shape[0], size=(filter_openset_size,), replace=False)
    print(simscore_idx)
    # print('tau: {}, set size: {}, element match (max freq: {}, min freq: {})'.format(tau, simscore_idx.shape, simscore_temp.max(), simscore_temp.min()))
    coreset_code = openset_code_filter[simscore_idx]
    coreset_path = openset_path_filter[simscore_idx]
    print('Coreset size: {}'.format(coreset_code.shape))
    
    # mask_array = np.zeros(shape=(1, openset_code_filter.shape[0]), dtype=np.int32)
    # mask_array[0, simscore_idx] = 1
    # mask_array = np.tile(mask_array, (downstream_code.shape[0], 1))

    # # Save affinity matrix plot
    # print('Calculating affinity matrix for bloom samples after filtering.....')
    # mask_affinity_mat = affinity_mat * mask_array
    # plt.figure(figsize=(15, 10))
    # plt.imshow(mask_affinity_mat, cmap='viridis', aspect=str(openset_code_filter.shape[0]/downstream_code.shape[0]))
    # plt.colorbar(shrink=1.0)
    # plt.grid(False)
    # plt.title('Bloom Samples After Filtering', fontsize=25)
    # plt.xlabel('Openset Bloom Data', fontsize=25)
    # plt.ylabel('Downstream Data', fontsize=25)
    # # plt.yticks(ticks=np.linspace(1, downstream_code.shape[0], 200))
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.tight_layout()
    # plt.savefig('data/AffinityMat_max/{}/downstream_{}_codelength_{}_hash_bloomfilter_{}.pdf'.format(downstream_data_name, downstream_data_name, code_length, openset_name), bbox_inches='tight', dpi=300)
    # plt.clf()
    # plt.close('all')
    # #####


    print('Building K-Means clustering.....')
    label_with_freq = {}
    path_with_freq  = []
    all_code  = np.concatenate([downstream_code, coreset_code], axis=0)
    all_paths = np.concatenate([downstream_path, coreset_path], axis=0)

    klabel, kcenter = KMeans(n_clusters=n_clusters).fit_predict(torch.from_numpy(all_code).to('cuda'))
    print(all_code.shape, all_paths.shape, klabel.shape, kcenter.shape)
    
    klabel = klabel.detach().cpu().numpy()

    for label in klabel:
        if label in label_with_freq:
            label_with_freq[label] += 1
        else:
            label_with_freq[label] = 1
    
    total_class         = len(label_with_freq.keys())
    avg_count_per_class = sum(label_with_freq.values())/len(label_with_freq.values())
    max_count_class     = max(label_with_freq.values())
    min_count_class     = min(label_with_freq.values())
    print('Total classes: {}\nAvg. sample per class: {}\nMax. sample in class: {}\nMin. sample in class: {}'.format(total_class, avg_count_per_class, max_count_class, min_count_class))

    os.makedirs('data/Filter/{}'.format(downstream_data_name), exist_ok=True)

    with open('data/Filter/{}/downopen_max_{}_subset_codelength_{}_hash_k{}_{}_kmeans.txt'.format(downstream_data_name, openset_name ,code_length, n_clusters, downstream_data_name), 'w') as file:
        for label, path in zip(klabel, all_paths):
            path_with_freq.append([path, label_with_freq[label]])
            file.write('{}: {}\n'.format(path, label_with_freq[label]))
    
    path_with_freq = np.array(path_with_freq)
    
    np.save('data/Filter/{}/downopen_max_ssl_filter_counting_{}_{}_fast_k{}_with_freq_clip_codelength_{}_{}_kmeans.npy'.format(downstream_data_name, openset_name, downstream_data_name, n_clusters, code_length, openset_name), path_with_freq)

    print('Total time: {}mins'.format((time()-start_time)/60.0 ))