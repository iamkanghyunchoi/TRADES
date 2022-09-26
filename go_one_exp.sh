#!/bin/bash

#SBATCH -J trades
#SBATCH -o out.trades.%j
#SBATCH --qos=big_qos
#SBATCH --partition=big
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

echo $CUDA_VISIBLE_DEVICES

cd ~
cd TRADES

source venv/bin/activate

python3 train_trades_cifar10_wrn28.py --eps_score high --model-dir ./model-cifar-wideResNet-scratch
