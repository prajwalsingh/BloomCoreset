import os
import cv2
import json
import argparse
import numpy as np
from glob import glob
from natsort import natsorted
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

np.random.seed(45)
# style.use('seaborn')
# plt.rcParams.update({'font.size': 70})

def parse_args():
    parser = argparse.ArgumentParser('argument for coreset sampling')

    parser.add_argument('--downstream', type=str, required=True, help='downstream dataset name')
    parser.add_argument('--path', type=str, required=True, help='openset path txt file')
    parser.add_argument('--path_label', type=str, help='openset path json for class labels')
    # parser.add_argument('--openset', type=str, required=True, help='openset dataset name')
    # parser.add_argument('--dataset_loc', type=str, default='../dataset', help='openset dataset name')
    # parser.add_argument('--percent', type=float, default=0.01, help='openset dataset name')
    # parser.add_argument('--clip_batch', type=int, default=512, help='batch size for inferencing clip encoding')
    # parser.add_argument('--clip_workers', type=int, default=16, help='number of workers for inferencing clip encoding')
    # parser.add_argument('--with_freq', type=bool, default=True, help='store filter data with frequency count')
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    txtfile_path = args.path
    downstream = args.downstream

    os.makedirs('save/plots/{}'.format(downstream), exist_ok=True)

    with open(txtfile_path, 'r') as file:
        data = file.readlines()
        data = list(map(str.strip, data))
        data = [path for path in data if 'imagenet1k' in path]
    
    with open(args.path_label) as json_file:
        class_labels = json.load(json_file)
        class_labels = dict(class_labels.values())
    
    class_data = [path.split(os.path.sep)[-2] for path in data]
    class_data_table  = {}
    for key in class_data:
        if key in class_data_table:
            class_data_table[key] += 1
        else:
            class_data_table[key] = 1
    
    class_data_table = dict(sorted(class_data_table.items(), key=lambda x:x[1], reverse=True))

    path_dict = {}
    for path in data:
        key  = path.split(os.path.sep)[-2]
        path = path.split(':')[0]
        if key in path_dict:
            path_dict[key].append(path)
        else:
            path_dict[key] = [path]

    total_class = len(class_data_table.keys())
    avg_count_per_class = sum(class_data_table.values())/len(class_data_table.values())
    max_count_class = max(class_data_table.values())
    min_count_class = min(class_data_table.values())
    print('Total classes: {}\nAvg. sample per class: {}\nMax. sample in class: {}\nMin. sample in class: {}'.format(total_class, avg_count_per_class, max_count_class, min_count_class))

    topk = total_class

    class_data_table = np.array([list(item) for item in class_data_table.items()])

    class_data_table_labels = [class_labels[key] for key in class_data_table[:, 0]]

    # All image
    plt.bar(np.int32(list(range(0, topk))), np.int32(class_data_table[:topk, 1]), alpha=0.5)
    plt.grid(False)
    plt.xlabel('All Classes')
    plt.ylabel('Count')
    plt.tight_layout()
    # plt.show()
    plt.savefig('save/plots/{}/all_plot_imagenet_{}_kmeans_fv2.pdf'.format(downstream, downstream), bbox_inches='tight', dpi=300)
    plt.close()

    topk = 20

    # Top-20 image
    # plt.bar(class_data_table_labels[:topk], np.int32(class_data_table[:topk, 1]), alpha=0.7)
    bars = plt.bar(np.int32(list(range(1, topk+1))), np.int32(class_data_table[:topk, 1]), alpha=0.9, lw=0.8, edgecolor='black')
    for idx, bar in enumerate(bars):
        bar.set_color('mediumslateblue')
        bar.set_edgecolor('black')
    # plt.bar_label(bars, label_type='center', rotation=90, c='white', fontsize=12)
    for x,y,text in zip(range(1, topk+1), np.int32(class_data_table[:topk, 1]), class_data_table_labels[:topk]):
        text = ' '.join(text.lower().split('_'))
        plt.text(x, y, text, rotation=90, ha='center', va='top', color='white', fontsize=17)
    plt.grid(False)
    # plt.xticks(ticks=list(range(0, topk)))
    plt.title('Imagenet Top-20 Sampled Classes', fontsize=19)
    # plt.xlabel('Top-K Most Frequent Classes', fontsize=14)
    # plt.ylabel('Frequency', fontsize=14)
    plt.tight_layout()
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.gca().yaxis.set_major_locator(MultipleLocator(20))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(5))
    plt.gca().xaxis.set_major_locator(MultipleLocator(5))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
    # plt.show()
    plt.savefig('save/plots/{}/topk_plot_imagenet_{}_kmeans_fv2.pdf'.format(downstream, downstream), bbox_inches='tight', dpi=300)
    plt.close()

    W = 5
    H = topk//W

    topk_imagelst = []

    # Top-1 image
    for key in class_data_table[:topk, 0]:
        topk_imagelst.append(cv2.resize(cv2.cvtColor(cv2.imread(path_dict[key][0], 1), cv2.COLOR_BGR2RGB), (224, 224)))
    
    plt.figure(figsize=(10,10))
    for i in range(H*W):
        plt.subplot(H,W,i+1)    # the number of images in the grid is 5*5 (25)
        ax = plt.gca()
        ax.imshow(topk_imagelst[i])
        ax.grid(False)
        ax.axis('off')
        ax.set_aspect('equal')
    plt.tight_layout()
    # plt.show()
    plt.subplots_adjust(wspace=0.05, hspace=-0.5)
    plt.savefig('save/plots/{}/topk_images_imagenet_{}_kmeans_fv2.pdf'.format(downstream, downstream), bbox_inches='tight', dpi=300)
    plt.close()

    bottomk_imagelst = []
    # Top-1 image
    for key in class_data_table[total_class-topk-1:, 0]:
        bottomk_imagelst.append(cv2.resize(cv2.cvtColor(cv2.imread(path_dict[key][0], 1), cv2.COLOR_BGR2RGB), (224, 224)))
    
    plt.figure(figsize=(10,10))
    for i in range(H*W):
        plt.subplot(H,W,i+1)    # the number of images in the grid is 5*5 (25)
        ax = plt.gca()
        ax.imshow(bottomk_imagelst[i])
        ax.grid(False)
        ax.axis('off')
        ax.set_aspect('equal')
    plt.tight_layout()
    # plt.show()
    plt.subplots_adjust(wspace=0.05, hspace=-0.5)
    plt.savefig('save/plots/{}/bottomk_images_imagenet_{}_kmeans_fv2.pdf'.format(downstream, downstream), bbox_inches='tight', dpi=300)
    plt.close()