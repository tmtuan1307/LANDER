#!/bin/bash

# IID
python main.py --group=c100t5 --exp_name=lander_b0 --dataset cifar100 --method=lander --tasks=5 --num_users 5 --beta=0

# NIID (beta=1)
python main.py --group=c100t5 --exp_name=lander_b1 --dataset cifar100 --method=lander --tasks=5 --num_users 5 --beta=1

# NIID (beta=0.5)
python main.py --group=c100t5 --exp_name=lander_b05 --dataset cifar100 --method=lander --tasks=5 --num_users 5 --beta=0.5

# NIID (beta=0.1)
python main.py --group=c100t5 --exp_name=lander_b01 --dataset cifar100 --method=lander --tasks=5 --num_users 5 --beta=0.1