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
    --data_path_file ./data/Filter/pets/downstream_ssl_filter_counting_imagenet_pets_fast_k145_with_freq_clip_codelength_512_imagenet.npy \
    --method byol \
    --epochs 5000 \
    --optimizer adam \
    --learning_rate 1e-3 \
    --weight_decay 1e-6 \
    --balance_fact 0.0 \
    --with_freq True \
    --resume ./save/pets_resnet50_pretrain_byol_pets/epoch_500.pth

python train_sup.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/pets_resnet50_pretrain_byol_pets/last.pth \
    --method byol \
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
    --data_path_file ./data/Filter/cub/downstream_ssl_filter_counting_imagenet_cub_fast_k182_with_freq_clip_codelength_512_imagenet.npy \
    --method byol \
    --epochs 5000 \
    --optimizer adam \
    --learning_rate 1e-3 \
    --weight_decay 1e-6 \
    --balance_fact 0.0 \
    --with_freq True \

python train_sup.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/cub_resnet50_pretrain_byol_cub/last.pth \
    --method byol \
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
    --data_path_file ./data/Filter/aircraft/downstream_ssl_filter_counting_imagenet_aircraft_fast_k483_with_freq_clip_codelength_512_imagenet.npy \
    --method byol \
    --epochs 5000 \
    --optimizer adam \
    --learning_rate 1e-3 \
    --weight_decay 1e-6 \
    --balance_fact 0.0 \
    --with_freq True \

python train_sup.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/aircraft_resnet50_pretrain_byol_aircraft/last.pth \
    --method byol \
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
    --data_path_file ./data/Filter/cars/downstream_ssl_filter_counting_imagenet_cars_fast_k100_with_freq_clip_codelength_512_imagenet.npy \
    --method byol \
    --epochs 5000 \
    --optimizer adam \
    --learning_rate 1e-3 \
    --weight_decay 1e-6 \
    --balance_fact 0.0 \
    --with_freq True \

python train_sup.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/cars_resnet50_pretrain_byol_cars/last.pth \
    --method byol \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0