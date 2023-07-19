#!/usr/bin/env sh

root_path=$PWD
PY_FILE_PATH="$root_path/run_pretraining.py"

tensorboard_path="$root_path/tensorboard"
log_path="$root_path/exp_log"
ckpt_path="$root_path/ckpt"

mkdir -p $tensorboard_path
mkdir -p $log_path
mkdir -p $ckpt_path

export PYTHONPATH=$PWD

num_gpus="1"
export CUDA_VISIBLE_DEVICES="1"


#Distillation phase1
# python -m torch.distributed.launch --nproc_per_node=${num_gpus} --master_port=25251 run_pretrain_ddp.py \
# RANK=1 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=25251 
# gdb -args 
python run_pretrain_ddp.py \
                --local_rank 0 \
                --lr 6e-4 \
                --train_micro_batch_size_per_gpu 190 \
                --eval_micro_batch_size_per_gpu 20 \
                --gradient_accumulation_steps 1 \
                --mlm_model_type bert \
                --epoch 15 \
                --max_grad_norm 1.0 \
                --data_path_prefix /data1/yehua.zhang/wudao_h5_new  \
                --eval_data_path_prefix /data1/yehua.zhang/wudao_h5_new_eval \
                --tokenizer_path /data1/nlp/models/chinese-roberta-wwm-ext-large \
                --bert_config /data1/yang.zhao/pretrain_yutian/config_yehua.json \
                --tensorboard_path $tensorboard_path \
                --log_path $log_path \
                --ckpt_path $ckpt_path \
                --log_interval 100 \
                --wandb \
                --dtr
                # --checkpoint_activations \
                # --load_pretrain_model /data1/nlp/models/test_model_yutian/pytorch_model.bin \
                # --student_load_pretrain_model /data1/nlp/models/bert-small/pytorch_model.bin \
                # --bert_config /data1/nlp/models/chinese-roberta-wwm-ext-large/config.json \
