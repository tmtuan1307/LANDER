#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Tue Oct 10 2023 16:30:58 GMT+1100 (Australian Eastern Daylight Time)

wandb online
rm -rf run/*
python main.py --wandb=1 --group=5tasks_cifar100 --method=nayeravg --r=1e-2 --tasks=5  --beta=0 --nums=8000 --kd=1 \
--exp_name=nayeravg_c100_fr1_ln20r1e1bn1e3g40 --type -1 --bn 1e-3 --num_users 5 --fr 1 --syn_round 10 --swp 0 --lte_norm 10 \
--com_round 25 --g_steps 40