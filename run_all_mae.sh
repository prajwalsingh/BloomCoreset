########################################### Pets #########################################
#############################################################################################

TAG="pets"
DATA="pets"

python train_selfsup.py --tag $TAG \
    --no_sampling \
    --model mae_vit_base \
    --batch_size 128 \
    --precision \
    --dataset $DATA \
    --data_path_file ./data/Filter/pets/downstream_ssl_filter_counting_imagenet_pets_fast_k145_with_freq_clip_codelength_512_imagenet.npy \
    --method mae \
    --epochs 5000 \
    --cosine \
    --optimizer adamw \
    --learning_rate 3e-4 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --warm_epochs 100 \
    --balance_fact 0.0 \
    --with_freq True \
    --resume ./save/pets_mae_vit_base_pretrain_mae_pets/epoch_500.pth

python train_sup.py --tag $TAG \
    --dataset $DATA \
    --model mae_vit_base \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/pets_mae_vit_base_pretrain_mae_pets/last.pth \
    --method mae \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0


########################################### Birds #########################################
#############################################################################################

TAG="cub"
DATA="cub"

python train_selfsup.py --tag $TAG \
    --no_sampling \
    --model mae_vit_base \
    --batch_size 128 \
    --precision \
    --dataset $DATA \
    --data_path_file ./data/Filter/cub/downstream_ssl_filter_counting_imagenet_cub_fast_k182_with_freq_clip_codelength_512_imagenet.npy \
    --method mae \
    --epochs 5000 \
    --cosine \
    --optimizer adamw \
    --learning_rate 3e-4 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --warm_epochs 100 \
    --balance_fact 0.0 \
    --with_freq True \

python train_sup.py --tag $TAG \
    --dataset $DATA \
    --model mae_vit_base \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/cub_mae_vit_base_pretrain_mae_cub/last.pth \
    --method mae \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0


########################################### Aircraft #########################################
#############################################################################################

TAG="aircraft"
DATA="aircraft"

python train_selfsup.py --tag $TAG \
    --no_sampling \
    --model mae_vit_base \
    --batch_size 128 \
    --precision \
    --dataset $DATA \
    --data_path_file ./data/Filter/aircraft/downstream_ssl_filter_counting_imagenet_aircraft_fast_k483_with_freq_clip_codelength_512_imagenet.npy \
    --method mae \
    --epochs 5000 \
    --cosine \
    --optimizer adamw \
    --learning_rate 3e-4 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --warm_epochs 100 \
    --balance_fact 0.0 \
    --with_freq True \

python train_sup.py --tag $TAG \
    --dataset $DATA \
    --model mae_vit_base \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/aircraft_mae_vit_base_pretrain_mae_aircraft/last.pth \
    --method mae \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0


########################################### Cars #########################################
#############################################################################################

TAG="cars"
DATA="cars"

python train_selfsup.py --tag $TAG \
    --no_sampling \
    --model mae_vit_base \
    --batch_size 128 \
    --precision \
    --dataset $DATA \
    --data_path_file ./data/Filter/cars/downstream_ssl_filter_counting_imagenet_cars_fast_k100_with_freq_clip_codelength_512_imagenet.npy \
    --method mae \
    --epochs 5000 \
    --cosine \
    --optimizer adamw \
    --learning_rate 3e-4 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --warm_epochs 100 \
    --balance_fact 0.0 \
    --with_freq True \

python train_sup.py --tag $TAG \
    --dataset $DATA \
    --model mae_vit_base \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/cars_mae_vit_base_pretrain_mae_cars/last.pth \
    --method mae \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0