import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patches as mpatches
from matplotlib.container import BarContainer
from matplotlib.container import Container
from matplotlib.artist import Artist
import matplotlib.patches
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

np.random.seed(45)
# style.use('seaborn')
# plt.rcParams.update({'font.size': 70})

if __name__ == '__main__':

    # dataset = ['w/o OS', 'COCO', 'iNaturalist', 'Places365', 'ImageNet']
    # dataset_idx = {0:'Pets', 1:'Action'}
    # baseline = {'Pets': [59.23, 0.0, 64.53, 63.18, 0.0, 64.73, 100.0, 0.0, 65.23, 100.0, 0.0, 79.63, 100.0],
    #             'Action': [43.76, 0.0, 62.66, 100.0, 0.0, 52.76, 100.0, 0.0, 65.26, 100.0, 0.0, 67.46, 100.0]}
    # colors = ['grey', 'white', 'darksalmon', 'darksalmon', 'white', 'seagreen', 'seagreen', 'white', 'mediumslateblue', 'mediumslateblue', 'white', 'teal', 'teal']
    # patterns = ['', '', '', 'x', '', '', 'x', '', '', 'x', '', '', 'x']
    # dataset = ['w/o OS', 'COCO', 'iNaturalist', 'ImageNet']
    # dataset_idx = {0:'Pets', 1:'Birds', 2:'Action', 3:'Indoor'}
    # baseline = {'Pets': [59.23, 0.0, 64.53, 63.18, 0.0, 64.73, 100.0, 0.0, 79.63, 76.20],
    #             'Birds': [29.27, 0.0, 30.87, 100.0, 0.0, 39.17, 100.0, 0.0, 37.67, 35.96],
    #             'Action': [43.76, 0.0, 62.66, 100.0, 0.0, 52.76, 100.0, 0.0, 67.46, 56.76],
    #             'Indoor': [54.10, 0.0, 65.4, 100.0, 0.0, 57.3, 100.0, 0.0, 72.01, 63.66]}
    # # Stanford40: Action, Mit67: Indoor
    # colors = ['grey', 'white', 'darksalmon', 'darksalmon', 'white', 'seagreen', 'seagreen', 'white', 'teal', 'teal']
    # patterns = ['', '', '', '/', '', '', '/', '', '', '/']
    # fontcolor = ['black', 'white', 'black', 'black', 'white', 'black', 'black', 'white', 'black', 'black', 'white', 'black', 'black']
    dataset = ['w/o OS', 'COCO', 'iNaturalist', 'ImageNet']
    dataset_idx = {0:'Pets', 1:'Birds', 2:'Action', 3:'Indoor'}
    baseline = {'Pets': [59.23,  65.65, 63.18,  64.73, 64.64,  76.90, 76.20],
                'Birds': [29.27,  27.93, 31.97,  39.17, 33.99,  36.94, 35.96],
                'Action': [43.76,  44.74, 45.82,  52.76, 45.73,  52.98, 56.76],
                'Indoor': [54.10, 55.42, 56.55,  57.3, 54.78,  59.18, 63.66]}
    # Stanford40: Action, Mit67: Indoor
    colors = ['grey', 'darksalmon', 'darksalmon', 'seagreen', 'seagreen', 'teal', 'teal']
    patterns = ['', '', 'x', '', 'x', '', 'x']

    H, W = 2, 2
    
    # plt.figure(figsize=(8, 8))
    fig, _ = plt.subplots(H,W, figsize=(7, 5))

    for i in range(H*W):
        plt.subplot(H,W,i+1)    # the number of images in the grid is 5*5 (25)
        ax = plt.gca()
        # ax.margins(x=0)
        # ax.margins(y=0)
        y_data = baseline[dataset_idx[i]]
        x_data = list(range(len(y_data)))
        bars = ax.bar(x_data, y_data, alpha=0.8, width=0.95, hatch=patterns, align='center', edgecolor='black', lw=.01)
        for idx, bar in enumerate(bars):
            bar.set_color(colors[idx])
            bar.set_edgecolor('black')
        bars.remove_callback(1)
        # for bar in bars:
        #     filter_bars = 
        # print(bars)
        # for bar in bars:
        #     print(bar)
        #     ax.bar_label(matplotlib.patches(bar), label_type='center', rotation=90)
        ax.bar_label(bars, label_type='center', rotation=90, c='white', fontsize=12)
        ax.get_xaxis().set_visible(False)
        ax.grid(False)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        plt.yticks(fontsize=12)
        # ax.axis('off')
        # ax.set_aspect('equal')
        ax.set_title('{}'.format(dataset_idx[i]), loc='center', fontsize=16)

    plt.tight_layout()

    plt.legend(handles=[mpatches.Patch(color='grey', label='w/o OS'),
                        mpatches.Patch(color='darksalmon', label='COCO'),
                        mpatches.Patch(color='seagreen', label='iNaturalist'),
                        # mpatches.Patch(color='mediumslateblue', label='Places365'),
                        mpatches.Patch(color='teal', label='ImageNet'),
                        mpatches.Patch(edgecolor=[0, 0, 0], label='Ours', hatch ='x', linewidth=0),
                        ], ncol=6, loc=(-1.55, -0.25), fontsize=12)

    # fig.legend(dataset, loc=(0.4, 0.85), ncols=5)
    # plt.show()
    # plt.subplots_adjust(wspace=0.2q, hspace=0.5)
    # plt.show()
    plt.savefig('save_cvpr/plots/comparison.pdf', bbox_inches='tight', dpi=300)
    plt.close()