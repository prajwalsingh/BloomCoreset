# python generate_coreset.py --downstream='pets' --openset='coco' --percent=0.01 --dataset_loc='../dataset' --clip_batch=512 --with_freq=True
# python generate_coreset.py --downstream='pets' --openset='inaturalistmini' --percent=0.01 --dataset_loc='../dataset' --clip_batch=512 --with_freq=True
python generate_coreset.py --downstream='pets' --openset='places365' --percent=0.01 --dataset_loc='../dataset' --clip_batch=512 --with_freq=True
python generate_coreset.py --downstream='pets' --openset='all' --percent=0.01 --dataset_loc='../dataset' --clip_batch=512 --with_freq=True

# python generate_coreset.py --downstream='cub' --openset='coco' --percent=0.01 --dataset_loc='../dataset' --clip_batch=512 --with_freq=True
# python generate_coreset.py --downstream='cub' --openset='inaturalistmini' --percent=0.01 --dataset_loc='../dataset' --clip_batch=512 --with_freq=True
python generate_coreset.py --downstream='cub' --openset='places365' --percent=0.01 --dataset_loc='../dataset' --clip_batch=512 --with_freq=True
python generate_coreset.py --downstream='cub' --openset='all' --percent=0.01 --dataset_loc='../dataset' --clip_batch=512 --with_freq=True

# python generate_coreset.py --downstream='stanford40' --openset='coco' --percent=0.01 --dataset_loc='../dataset' --clip_batch=512 --with_freq=True
# python generate_coreset.py --downstream='stanford40' --openset='inaturalistmini' --percent=0.01 --dataset_loc='../dataset' --clip_batch=512 --with_freq=True
python generate_coreset.py --downstream='stanford40' --openset='places365' --percent=0.01 --dataset_loc='../dataset' --clip_batch=512 --with_freq=True
python generate_coreset.py --downstream='stanford40' --openset='all' --percent=0.01 --dataset_loc='../dataset' --clip_batch=512 --with_freq=True

# python generate_coreset.py --downstream='mit67' --openset='coco' --percent=0.01 --dataset_loc='../dataset' --clip_batch=512 --with_freq=True
# python generate_coreset.py --downstream='mit67' --openset='inaturalistmini' --percent=0.01 --dataset_loc='../dataset' --clip_batch=512 --with_freq=True
python generate_coreset.py --downstream='mit67' --openset='places365' --percent=0.01 --dataset_loc='../dataset' --clip_batch=512 --with_freq=True
python generate_coreset.py --downstream='mit67' --openset='all' --percent=0.01 --dataset_loc='../dataset' --clip_batch=512 --with_freq=True