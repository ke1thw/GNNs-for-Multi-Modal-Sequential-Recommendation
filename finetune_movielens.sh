MASTER_ADDRESS="127.0.0.1"
MASTER_PORT="12309"

torchrun \
    --nnodes=1:5 \
    --nproc_per_node=8 \
    --max_restarts=0 \
    --rdzv_id=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDRESS}:${MASTER_PORT}" \
    finetune_movielens.py \
    --config="./configs/finetuning/finetune_movielens.yaml"























