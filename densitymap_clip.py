import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torchvision
import torch
from tsnecuda import TSNE
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import open_clip
from data_list import ImageList
from tqdm import tqdm
from torch.autograd import Variable
import argparse

style.use('seaborn')
# plt.rcParams.update({'font.size': 70})


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
            is_start = False
        else:
            all_output = torch.cat((all_output, y.data.cpu().float()), 0)

    return all_output.cpu().numpy()

def parse_args():
    parser = argparse.ArgumentParser('argument for coreset sampling')

    parser.add_argument('--name', type=str, required=True, help='downstream dataset name')
    parser.add_argument('--downstream', type=str, required=True, help='sampled downstream path txt location')
    parser.add_argument('--openset', type=str, required=True, help='sampled openset path txt location')
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    downstream_txtfile = args.downstream
    openset_txtfile    = args.openset
    downstream = args.name

    os.makedirs('save/plots/{}'.format(downstream), exist_ok=True)

    batch_size = 512
    workers = 8
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to('cuda')

    with open(downstream_txtfile, 'r') as file:
        imagelists = file.readlines()
        imagelists = list(map(str.strip, imagelists))
    
    imagelists = [path.split(':')[0] for path in imagelists]

    database = ImageList(imagelists, preprocess=preprocess, ret_path=True)    
    data_loader = torch.utils.data.DataLoader(database, batch_size = batch_size, shuffle=False, num_workers=workers)

    # Assuming features_df is your DataFrame containing feature vectors
    # If you want to plot a subset of features, you can select those features here
    high_dim_data = predict_hash_code(model, data_loader)

    tsne = TSNE(n_components=2, perplexity=40, learning_rate=10.0, n_iter=2000, random_seed=42)
    tsne_embeddings = tsne.fit_transform(high_dim_data)

    tsne_embeddings /= np.linalg.norm(tsne_embeddings, axis=1)[:, np.newaxis]

    # Fit a Gaussian KDE to the normalized t-SNE embeddings
    kde = multivariate_normal(mean=[0, 0], cov=0.1)  # Adjust the covariance as needed
    density_values = kde.pdf(tsne_embeddings)
    
    # kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(tsne_embeddings)
    # density_values = kde.score_samples(tsne_embeddings)


    # print(density_values.shape)

    # Plot the unit ring with KDE distribution
    plt.figure(figsize=(5, 5))
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=density_values, cmap='summer', alpha=0.01, s=200)
    # plt.colorbar()
    # plt.title('t-SNE Embeddings with Gaussian Kernel Density Estimation')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.show()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('save/plots/{}/clip_densityplot_downstream{}.pdf'.format(downstream, downstream), bbox_inches='tight', dpi=300)
    plt.close()


    with open(openset_txtfile, 'r') as file:
        imagelists = file.readlines()
        imagelists = list(map(str.strip, imagelists))
    
    # class_data = [path.split(os.path.sep)[-2] for path in imagelists]
    # class_data_table  = {}
    # for key in class_data:
    #     if key in class_data_table:
    #         class_data_table[key] += 1
    #     else:
    #         class_data_table[key] = 1
    
    # class_data_table = dict(sorted(class_data_table.items(), key=lambda x:x[1], reverse=True))

    # path_dict = {}
    # for path in imagelists:
    #     key  = path.split(os.path.sep)[-2]
    #     path = path.split(':')[0]
    #     if key in path_dict:
    #         path_dict[key].append(path)
    #     else:
    #         path_dict[key] = [path]
    
    imagelists = [path.split(':')[0] for path in imagelists]

    # imagelists = []
    # key_lst = list(class_data_table.keys())

    # for key in key_lst[:200]:
    #     imagelists.extend(path_dict[key])

    database = ImageList(imagelists, preprocess=preprocess, ret_path=True)    
    data_loader = torch.utils.data.DataLoader(database, batch_size = batch_size, shuffle=False, num_workers=workers)

    # Assuming features_df is your DataFrame containing feature vectors
    # If you want to plot a subset of features, you can select those features here
    high_dim_data = predict_hash_code(model, data_loader)

    tsne = TSNE(n_components=2, perplexity=40, learning_rate=10.0, n_iter=2000, random_seed=42)
    tsne_embeddings = tsne.fit_transform(high_dim_data)

    tsne_embeddings /= np.linalg.norm(tsne_embeddings, axis=1)[:, np.newaxis]

    # Fit a Gaussian KDE to the normalized t-SNE embeddings
    kde = multivariate_normal(mean=[0, 0], cov=0.1)  # Adjust the covariance as needed
    density_values = kde.pdf(tsne_embeddings)
    # kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(tsne_embeddings)
    # density_values = kde.score_samples(tsne_embeddings)

    # print(density_values.shape)

    # Plot the unit ring with KDE distribution
    plt.figure(figsize=(5, 5))
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=density_values, cmap='summer', alpha=0.01, s=200)
    # plt.colorbar()
    # plt.title('t-SNE Embeddings with Gaussian Kernel Density Estimation')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    plt.grid(False)
    plt.tight_layout()
    # plt.show()
    plt.savefig('save/plots/{}/clip_densityplot_coreset{}.pdf'.format(downstream, downstream), bbox_inches='tight', dpi=300)
    plt.close()