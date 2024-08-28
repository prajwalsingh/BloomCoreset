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
    --resume ./save/celeba_resnet50_pretrain_simclr_celeba/epoch_4000.pth

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