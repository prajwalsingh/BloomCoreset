########################################### Indoor #########################################
#############################################################################################

TAG="mit67"
DATA="mit67"

python train_selfsup.py --tag $TAG \
    --no_sampling \
    --model resnet50 \
    --batch_size 256 \
    --precision \
    --dataset $DATA \
    --data_path_file ./data/Filter/mit67/downopen_max_ssl_filter_counting_imagenet_mit67_fast_k2048_with_freq_clip_codelength_512_imagenet_kmeans.npy \
    --method simclr \
    --epochs 5000 \
    --cosine \
    --optimizer sgd \
    --cosine \
    --learning_rate 1e-1 \
    --weight_decay 1e-4 \
    --balance_fact 0.0 \
    --with_freq True \

python train_sup.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/mit67_resnet50_pretrain_simclr_mit67/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0


########################################### Birds #########################################
#############################################################################################

TAG="cub"
DATA="cub"

python train_selfsup.py --tag $TAG \
    --no_sampling \
    --model resnet50 \
    --batch_size 256 \
    --precision \
    --dataset $DATA \
    --data_path_file ./data/Filter/cub/downopen_max_ssl_filter_counting_imagenet_cub_fast_k2048_with_freq_clip_codelength_512_imagenet_kmeans.npy \
    --method simclr \
    --epochs 5000 \
    --cosine \
    --optimizer sgd \
    --cosine \
    --learning_rate 1e-1 \
    --weight_decay 1e-4 \
    --balance_fact 0.0 \
    --with_freq True \

python train_sup.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/cub_resnet50_pretrain_simclr_cub/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0


########################################### Aircraft #########################################
#############################################################################################

TAG="aircraft"
DATA="aircraft"

python train_selfsup.py --tag $TAG \
    --no_sampling \
    --model resnet50 \
    --batch_size 256 \
    --precision \
    --dataset $DATA \
    --data_path_file ./data/Filter/aircraft/downopen_max_ssl_filter_counting_imagenet_aircraft_fast_k2048_with_freq_clip_codelength_512_imagenet_kmeans.npy \
    --method simclr \
    --epochs 5000 \
    --cosine \
    --optimizer sgd \
    --cosine \
    --learning_rate 1e-1 \
    --weight_decay 1e-4 \
    --balance_fact 0.0 \
    --with_freq True \

python train_sup.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/aircraft_resnet50_pretrain_simclr_aircraft/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0


########################################### Cars #########################################
#############################################################################################

TAG="cars"
DATA="cars"

python train_selfsup.py --tag $TAG \
    --no_sampling \
    --model resnet50 \
    --batch_size 256 \
    --precision \
    --dataset $DATA \
    --data_path_file ./data/Filter/cars/downopen_max_ssl_filter_counting_imagenet_cars_fast_k2048_with_freq_clip_codelength_512_imagenet_kmeans.npy \
    --method simclr \
    --epochs 5000 \
    --cosine \
    --optimizer sgd \
    --cosine \
    --learning_rate 1e-1 \
    --weight_decay 1e-4 \
    --balance_fact 0.0 \
    --with_freq True \

python train_sup.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/cars_resnet50_pretrain_simclr_cars/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0

# ########################################### Dogs #########################################
# #############################################################################################

# TAG="dogs"
# DATA="dogs"

# python train_selfsup.py --tag $TAG \
#     --no_sampling \
#     --model resnet50 \
#     --batch_size 256 \
#     --precision \
#     --dataset $DATA \
#     --data_path_file ./data/Filter/dogs/downstream_max_ssl_filter_counting_inaturalistmini_dogs_fast_k19_with_freq_clip_codelength_512_inaturalistmini.npy \
#     --method simclr \
#     --epochs 5000 \
#     --cosine \
#     --optimizer sgd \
#     --cosine \
#     --learning_rate 1e-1 \
#     --weight_decay 1e-4 \
#     --balance_fact 0.0 \
#     --with_freq True \

# python train_sup.py --tag $TAG \
#     --dataset $DATA \
#     --model resnet50 \
#     --data_folder ../dataset/ \
#     --pretrained \
#     --pretrained_ckpt ./save/dogs_resnet50_pretrain_simclr_dogs/last.pth \
#     --method simclr \
#     --epochs 100 \
#     --learning_rate 10 \
#     --weight_decay 0


########################################### Food #########################################
#############################################################################################

TAG="food11"
DATA="food11"

python train_selfsup.py --tag $TAG \
    --no_sampling \
    --model resnet50 \
    --batch_size 256 \
    --precision \
    --dataset $DATA \
    --data_path_file ./data/Filter/food11/downopen_max_ssl_filter_counting_imagenet_food11_fast_k2048_with_freq_clip_codelength_512_imagenet_kmeans.npy \
    --method simclr \
    --epochs 5000 \
    --cosine \
    --optimizer sgd \
    --cosine \
    --learning_rate 1e-1 \
    --weight_decay 1e-4 \
    --balance_fact 0.0 \
    --with_freq True \

python train_sup.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/food11_resnet50_pretrain_simclr_food11/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0