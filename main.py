import argparse
import wandb, os
from utils.data_manager import DataManager, setup_seed
from utils.toolkit import count_parameters
from methods.finetune import Finetune
from methods.icarl import iCaRL
from methods.lwf import LwF
from methods.ewc import EWC
from methods.target import TARGET
from methods.lander import LANDER
import warnings

warnings.filterwarnings('ignore')


def get_learner(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        return iCaRL(args)
    elif name == "ewc":
        return EWC(args)
    elif name == "lwf":
        return LwF(args)
    elif name == "finetune":
        return Finetune(args)
    elif name == "target":
        return TARGET(args)
    elif name == "lander":
        return LANDER(args)
    else:
        assert 0


def train(args):
    setup_seed(args["seed"])
    # setup the dataset and labels
    data_manager = DataManager(
        args["dataset"],
        args["class_shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args
    )
    args["class_order"] = data_manager.get_class_order()
    learner = get_learner(args["method"], args)
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}

    # train for each task
    for task in range(data_manager.nb_tasks):
    # for task in range(2):
        print("All params: {}, Trainable params: {}".format(count_parameters(learner._network),
                                                            count_parameters(learner._network, True)))
        learner.incremental_train(data_manager)  # train for one task
        cnn_accy, nme_accy = learner.eval_task()
        learner.after_task()

        print("CNN: {}".format(cnn_accy["grouped"]))
        cnn_curve["top1"].append(cnn_accy["top1"])
        print("CNN top1 curve: {}".format(cnn_curve["top1"]))


def args_parser():
    parser = argparse.ArgumentParser(description='benchmark for federated continual learning')
    # Exp settings
    parser.add_argument('--exp_name', type=str, default='lander_b0', help='name of this experiment')
    parser.add_argument('--wandb', type=int, default=0, help='1 for using wandb')
    parser.add_argument('--save_dir', type=str, default="", help='save the syn data')
    parser.add_argument('--project', type=str, default="LANDER", help='wandb project')
    parser.add_argument('--group', type=str, default="c100", help='wandb group')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--spec', type=str, default="t1", help='choose a model')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

    # federated continual learning settings
    parser.add_argument('--dataset', type=str, default="cifar100", help='which dataset')
    parser.add_argument('--tasks', type=int, default=5, help='num of tasks')
    parser.add_argument('--method', type=str, default="lander", help='choose a learner')
    parser.add_argument('--net', type=str, default="resnet18", help='choose a model')
    parser.add_argument('--com_round', type=int, default=100, help='communication rounds')
    parser.add_argument('--local_ep', type=int, default=2, help='local training epochs')
    parser.add_argument('--num_users', type=int, default=5, help='num of clients')
    parser.add_argument('--local_bs', type=int, default=128, help='local batch size')
    parser.add_argument('--beta', type=float, default=0.0, help='control the degree of label skew')
    parser.add_argument('--frac', type=float, default=1.0, help='the fraction of selected clients')
    parser.add_argument('--class_shuffle', type=int, default=1, help='class shuffle')

    # Data-free Generation
    parser.add_argument('--lr_g', default=2e-3, type=float, help='learning rate of generator')
    parser.add_argument('--synthesis_batch_size', default=256, type=int, help='synthetic data batch size')
    parser.add_argument('--bn', default=1.0, type=float, help='parameter for batchnorm regularization')
    parser.add_argument('--oh', default=0.5, type=float, help='parameter for similarity')
    parser.add_argument('--adv', default=1.0, type=float, help='parameter for diversity')
    parser.add_argument('--nz', default=256, type=int, help='output size of noisy nayer')
    parser.add_argument('--nums', type=int, default=10000, help='the num of synthetic data')
    parser.add_argument('--warmup', default=10, type=int, help='number of epoches generator only warmups not stores images')
    parser.add_argument('--syn_round', default=40, type=int, help='number of synthetize round.')
    parser.add_argument('--g_steps', default=40, type=int, help='number of generation steps.')

    # Client Training
    parser.add_argument('--num_worker', type=int, default=4, help='number of worker for dataloader')
    parser.add_argument('--mulc', type=str, default="fork", help='type of multi process for dataloader')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for optimizer')
    parser.add_argument('--syn_bs', default=1, type=int, help='number of old synthetic data in training, 1 for similar to local_bs')
    parser.add_argument('--local_lr', default=4e-2, type=float, help='learning rate for optimizer')

    # LANDER
    parser.add_argument('--r', default=0.015, type=float, help='LTE center radius')
    parser.add_argument('--ltc', default=5, type=float, help='lamda_ltc parameter for LTE center')
    parser.add_argument('--pre', type=float, default=0.4, help='alpha_pre for distilling from previous task')
    parser.add_argument('--cur', type=float, default=0.2, help='alpha_cur for current task training')

    parser.add_argument('--type', default=-1, type=int,
                        help='seed for initializing training.') # 0 for train forward, 1 pretrain stage 1, 2 pretrain stage 2
    parser.add_argument('--syn', default=1, type=int,
                        help='seed for initializing training.')  # 0 for train forward, 1 pretrain stage 1, 2 pretrain stage 2
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = args_parser()
    if args.dataset == "tiny_imagenet":
        args.num_class = 200
    elif args.dataset == "cifar100":
        args.num_class = 100
    elif args.dataset == "imagenet":
        args.num_class = 1000
    args.init_cls = int(args.num_class / args.tasks)
    args.increment = args.init_cls

    print(args)

    args.exp_name = f"{args.beta}_{args.method}_{args.exp_name}"

    dir = "run"
    if not os.path.exists(dir):
        os.makedirs(dir)
    args.save_dir = os.path.join(dir, args.group + "_" + args.exp_name + "" + args.spec)

    if args.wandb == 1:
        wandb.init(config=args, project=args.project, group=args.group, name=args.exp_name)
        wandb.run.log_code(".")
    args = vars(args)

    train(args)
