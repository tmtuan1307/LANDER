## The source code for "Text-Enhanced Data-free Approach for Federated Class-Incremental Learning" accepted by CVPR 2024.
## Paper link: https://arxiv.org/abs/2403.14101

# Method
In this paper, we introduce LANDER (Label Text Centered Data-Free Knowledge Transfer) to address this issue by utilizing label text embeddings (LTE) produced by pretrained language models. Specifically, during the model training phase, our approach treats LTE as anchor points and constrains the feature embeddings of corresponding training samples around them, enriching the surrounding area with more meaningful information. In the DFKT phase, by using these LTE anchors, LANDER can synthesize more meaningful samples, thereby effectively addressing the forgetting problem. Additionally, instead of tightly constraining embeddings toward the anchor, the Bounding Loss is introduced to encourage sample embeddings to remain flexible within a defined radius. This approach preserves the natural differences in sample embeddings and mitigates the embedding overlap caused by heterogeneous federated settings. Extensive experiments conducted on CIFAR100, Tiny-ImageNet, and ImageNet demonstrate that LANDER significantly outperforms previous methods and achieves state-of-the-art performance in FCIL. 
 
![alt text](https://github.com/tmtuan1307/LANDER/blob/main/cvpr2024_lander_thumb.png)

# Reproducing
We test the code on RTX 4090 GPU with pytorch: 
```
torch==2.0.1
torchvision==0.15.2
```

## Baseline
Here, we provide a simple example for different methods. 
For example, for `cifar100-5tasks`, please run the following commands to test the model performance with non-IID (`$\beta=0.5$`) data.

```
#!/bin/bash
# method= ["finetue", "lwf", "ewc", "icarl", "target"]

CUDA_VISIBLE_DEVICES=0 python main.py --group=c100t5 --exp_name=$method_b05 --dataset cifar100 --method=$method --tasks=5 --num_users 5 --beta=0.5
```

### Ours
```
CUDA_VISIBLE_DEVICES=0 python main.py --group=c100t5 --exp_name=lander_b05 --dataset cifar100 --method=lander --tasks=5 --num_users 5 --beta=0.5
```

## Citation:
  ```
@inproceedings{lander,
  title={Text-enhanced data-free approach for federated class-incremental learning},
  author={Tran, Minh-Tuan and Le, Trung and Le, Xuan-May and Harandi, Mehrtash and Phung, Dinh},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23870--23880},
  year={2024}
}
  ```
