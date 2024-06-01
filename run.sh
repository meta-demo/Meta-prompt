
for ((i=0; i<5; i++))
do

python train.py  \
    --config_file configs/models/rn50_ep50.yaml \
    --dataset_config_file configs/datasets/coco.yaml \
    --input_size 224  \
    --lr 0.02   \
    --mlplr 0.0005 \
    --output_dir model/coco/model_r${i} \
    --max_epochs 10 \
    --device_id 2 \
    --n_ctx 16 \
    --pool_size 8 \
    --beta 1 \
    --gamma 0.1 \
    --prompt_key_init uniform

done