export WANDB_API_KEY=589d531f376ac4cfc8d713753b96052a1420b709
export http_proxy=http://oversea-squid2.ko.txyun:11080
export https_proxy=http://oversea-squid2.ko.txyun:11080
export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN before running}

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --num_processes=4 --main_process_port=29504 train.py \
    --max-train-steps=400000 \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="no" \
    --data-dir="/share/xutongda/ImageNeth5/data_256" \
    --output-dir="exps" \
    --batch-size=128 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-XL/1" \
    --checkpointing-steps=50000 \
    --vae-config="./configs/VTPB.yaml" \
    --bn-momentum=0.1 \
    --exp-name="sit-xl-vtpb-400k" \
    &> sit-xl-vtpb-400k.out &

