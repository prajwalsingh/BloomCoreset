# BloomCoreset
Official implementation of the work BloomSSL: Fast Fine-Grained Self-Supervised Learning using Bloom Filters

## Fine-Grained SSL Framework

<img src="results/bloomsslframework.png" width="100%"/> 
<img src="results/bloomfilter.png" width="100%"/> 

## Sampled Coresets from ImageNet1k Open-Set using BloomSSL

| Classes  | Top-20 | Frequency Plot |
|---|---|---|
| Cars  | <img src="results/topk_images_imagenet_cars_page-0001.jpg" width="400px" height="300px"/>  | <img src="results/topk_plot_imagenet_cars_page-0001.jpg" width="400px" height="300px"/> |
| Cubs  | <img src="results/topk_images_imagenet_cub_page-0001.jpg" width="400px" height="300px"/>  | <img src="results/topk_plot_imagenet_cub_page-0001.jpg" width="400px" height="300px"/> |
| Dogs  | <img src="results/topk_images_imagenet_dogs_page-0001.jpg" width="400px" height="300px"/>  | <img src="results/topk_plot_imagenet_dogs_page-0001.jpg" width="400px" height="300px"/> |
| Texture (dtd)  | <img src="results/topk_images_imagenet_dtd_page-0001.jpg" width="400px" height="300px"/>  | <img src="results/topk_plot_imagenet_dtd_page-0001.jpg" width="400px" height="300px"/> |
| Indoor (mit67)  | <img src="results/topk_images_imagenet_mit67_page-0001.jpg" width="400px" height="300px"/>  | <img src="results/topk_plot_imagenet_mit67_page-0001.jpg" width="400px" height="300px"/> |
| Pets  | <img src="results/topk_images_imagenet_pets_page-0001.jpg" width="400px" height="300px"/>  | <img src="results/topk_plot_imagenet_pets_page-0001.jpg" width="400px" height="300px"/> |
| Action (stanford40)  | <img src="results/topk_images_imagenet_stanford40_page-0001.jpg" width="400px" height="300px"/>  | <img src="results/topk_plot_imagenet_stanford40_page-0001.jpg" width="400px" height="300px"/> |

## Results

<img src="results/bloomsslresults.png" width="100%"/> 

## References
1. Code is heavily based on OpenSSL-SimCore [[link](https://github.com/sungnyun/openssl-simcore)]; we thank the authors for making the code publicly available.
