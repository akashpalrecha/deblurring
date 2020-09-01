#!/bin/bash
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 02-00:00
#SBATCH -p seas_gpu
#SBATCH --mem=32768
#SBATCH --gres=gpu:1
#SBATCH -o /n/home00/apalrecha/deblurring/experiments/EDSR_GOPRO_Large_edsr_full/slurm_out/edsr_train_%j.out
#SBATCH -e /n/home00/apalrecha/deblurring/experiments/EDSR_GOPRO_Large_edsr_full/slurm_out/edsr_train_%j.err

cd ~
source ~/.bashrc
conda activate deblurring
cd deblurring
echo ""
echo "---- Starting Training ----"
python main.py --model_name edsr --dataset "../datasets/GOPRO_Large/" --tag edsr_full --use_stats GOPRO --gpus=1 --n_feats=256 --n_resblocks 32 --crop_size 128 128 --batch_size 8 --max_epochs 200
echo ""
echo "---- Training Complete ----"