TAG="mit67"
DATA="mit67"

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

# knn (Table 7a)
# --knn \
# --topk 20 200

# semisup (Table 7b)
# --label_ratio 0.1 \
# --e2e