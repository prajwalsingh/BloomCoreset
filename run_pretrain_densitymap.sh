TAG="aircraft"
DATA="aircraft"
python densitymap_pretrain.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/aircraft_1p_clip_49.34/aircraft_resnet50_pretrain_simclr_aircraft/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0 \
    --name=aircraft \
    --openset=data/Filter/aircraft/openset_imagenet_subset_codelength_512_hash_k483_aircraft.txt \
    --downstream=data/Filter/aircraft/downstream_aircraft_codelength_512_hash_k483_imagenet.txt

TAG="cars"
DATA="cars"
python densitymap_pretrain.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/cars_512_1p_clip_52.08/cars_resnet50_pretrain_simclr_cars/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0 \
    --name=cars --openset=data/Filter/cars/openset_imagenet_subset_codelength_512_hash_k100_cars.txt --downstream=data/Filter/cars/downstream_cars_codelength_512_hash_k100_imagenet.txt


TAG="celeba"
DATA="celeba"
python densitymap_pretrain.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/celeba_512_1p_clip_55.46/celeba_resnet50_pretrain_simclr_celeba/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0 \
    --name=celeba --openset=data/Filter/celeba/openset_imagenet_subset_codelength_512_hash_k186_celeba.txt --downstream=data/Filter/celeba/downstream_celeba_codelength_512_hash_k186_imagenet.txt


TAG="cub"
DATA="cub"
python densitymap_pretrain.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/cub_1p_clip_35.96/cub_resnet50_pretrain_simclr_cub/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0 \
    --name=cub --openset=data/Filter/cub/openset_imagenet_subset_codelength_512_hash_k182_cub.txt --downstream=data/Filter/cub/downstream_cub_codelength_512_hash_k182_imagenet.txt


TAG="dogs"
DATA="dogs"
python densitymap_pretrain.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/dogs_512_1p_clip_54.43/dogs_resnet50_pretrain_simclr_dogs/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0 \
    --name=dogs --openset=data/Filter/dogs/openset_imagenet_subset_codelength_512_hash_k2_dogs.txt --downstream=data/Filter/dogs/downstream_dogs_codelength_512_hash_k2_imagenet.txt


TAG="dtd"
DATA="dtd"
python densitymap_pretrain.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/dtd_1p_clip_66.49/dtd_resnet50_pretrain_simclr_dtd/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0 \
    --name=dtd --openset=data/Filter/dtd/openset_imagenet_subset_codelength_512_hash_k35_dtd.txt --downstream=data/Filter/dtd/downstream_dtd_codelength_512_hash_k35_imagenet.txt


TAG="flowers"
DATA="flowers"
python densitymap_pretrain.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/flowers_1p_clip_85.56/flowers_resnet50_pretrain_simclr_flowers/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0 \
    --name=flowers --openset=data/Filter/flowers/openset_imagenet_subset_codelength_512_hash_k657_flowers.txt --downstream=data/Filter/flowers/downstream_flowers_codelength_512_hash_k657_imagenet.txt


TAG="food11"
DATA="food11"
python densitymap_pretrain.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/food11_512_1p_clip_90.20/food11_resnet50_pretrain_simclr_food11/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0 \
    --name=food11 --openset=data/Filter/food11/openset_imagenet_subset_codelength_512_hash_k150_food11.txt --downstream=data/Filter/food11/downstream_food11_codelength_512_hash_k150_imagenet.txt


TAG="mit67"
DATA="mit67"
python densitymap_pretrain.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/mit67_1p_clip_63.66/mit67_resnet50_pretrain_simclr_mit67/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0 \
    --name=mit67 --openset=data/Filter/mit67/openset_imagenet_subset_codelength_512_hash_k39_mit67.txt --downstream=data/Filter/mit67/downstream_mit67_codelength_512_hash_k39_imagenet.txt


TAG="pets"
DATA="pets"
python densitymap_pretrain.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/pets_512_1p_clip_76.2/pets_resnet50_pretrain_simclr_pets/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0 \
    --name=pets --openset=data/Filter/pets/openset_imagenet_subset_codelength_512_hash_k145_pets.txt --downstream=data/Filter/pets/downstream_pets_codelength_512_hash_k145_imagenet.txt


TAG="stanford40"
DATA="stanford40"
python densitymap_pretrain.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder ../dataset/ \
    --pretrained \
    --pretrained_ckpt ./save/stanford40_512_1p_clip_56.76/stanford40_resnet50_pretrain_simclr_stanford40/last.pth \
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0 \
    --name=stanford40 --openset=data/Filter/stanford40/openset_imagenet_subset_codelength_512_hash_k18_stanford40.txt --downstream=data/Filter/stanford40/downstream_stanford40_codelength_512_hash_k18_imagenet.txt