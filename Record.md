nohup accelerate launch --num_processes=4 --gpu_ids="0,1,2,3" evalvae.py \
        --seed=0 \
        --sample-dir="./samples" \
        --exp-name="eval-sdvae" \
        --dataset="/video_ssd/lpm/ImageNet/val" \
        --dataset-ref="/video_ssd/lpm/ImageNet/train" \
        --small 50000 \
        --vae-config="./configs/SDVAE.yaml" &> evalvae_SDVAE.out &
