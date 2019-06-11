#! /bin/sh

python main.py --dataset mnist --seed 1 --lr 1e-4 --beta1 0.9 --beta2 0.999 \
    --objective mnist_mod --model mnist_mod --batch_size 64 --z_dim 10 --max_iter 30000 \
    --beta 10 --image_size 28 --viz_name mnist_gamma100_z10
