import os
import cv2
import numpy as np

if __name__ == '__main__':

    dataset_path = np.load('data/downstream_ssl_filterdata.npy', allow_pickle=False).tolist()

    os.makedirs('data/sample_images', exist_ok=True)

    for idx, path in enumerate(dataset_path[::-1]):
        print(path)
        img = cv2.imread('data/'+path, 1)
        cv2.imwrite('data/sample_images/{}.png'.format(idx), img)