########################################## Flowers #########################################
############################################################################################

TAG="flowers"
DATA="flowers"

python train_selfsup.py --tag $TAG \
    --no_sampling \
    --model resnet50 \
    --batch_size 256 \
    --precision \
    --dataset $DATA \
    --data_path_file ./data/Filter/flowers/downopen_max_ssl_filter_counting_imagenet_flowers_fast_k2048_with_freq_clip_codelength_512_imagenet_kmeans.npy \
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
    --pretrained_ckpt ./save/flowers_resnet50_pretrain_simclr_flowers/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0

########################################### Pets #########################################
#############################################################################################

TAG="pets"
DATA="pets"

python train_selfsup.py --tag $TAG \
    --no_sampling \
    --model resnet50 \
    --batch_size 256 \
    --precision \
    --dataset $DATA \
    --data_path_file ./data/Filter/pets/downopen_max_ssl_filter_counting_imagenet_pets_fast_k2048_with_freq_clip_codelength_512_imagenet_kmeans.npy \
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
    --pretrained_ckpt ./save/pets_resnet50_pretrain_simclr_pets/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0


########################################### Textures #########################################
#############################################################################################

TAG="dtd"
DATA="dtd"

python train_selfsup.py --tag $TAG \
    --no_sampling \
    --model resnet50 \
    --batch_size 256 \
    --precision \
    --dataset $DATA \
    --data_path_file ./data/Filter/dtd/downopen_max_ssl_filter_counting_imagenet_dtd_fast_k2048_with_freq_clip_codelength_512_imagenet_kmeans.npy \
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
    --pretrained_ckpt ./save/dtd_resnet50_pretrain_simclr_dtd/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0


########################################### Action #########################################
#############################################################################################

TAG="stanford40"
DATA="stanford40"

python train_selfsup.py --tag $TAG \
    --no_sampling \
    --model resnet50 \
    --batch_size 256 \
    --precision \
    --dataset $DATA \
    --data_path_file ./data/Filter/stanford40/downopen_max_ssl_filter_counting_imagenet_stanford40_fast_k2048_with_freq_clip_codelength_512_imagenet_kmeans.npy \
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
    --pretrained_ckpt ./save/stanford40_resnet50_pretrain_simclr_stanford40/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0


########################################### Faces #########################################
#############################################################################################

TAG="celeba"
DATA="celeba"

python train_selfsup.py --tag $TAG \
    --no_sampling \
    --model resnet50 \
    --batch_size 256 \
    --precision \
    --dataset $DATA \
    --data_path_file ./data/Filter/celeba/downopen_max_ssl_filter_counting_imagenet_celeba_fast_k2048_with_freq_clip_codelength_512_imagenet_kmeans.npy \
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
    --pretrained_ckpt ./save/celeba_resnet50_pretrain_simclr_celeba/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0