from unit import UNIT
import argparse
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True, help="experiment name")
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="apple2orange", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator samples")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
    parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
    parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")

    return check_args(parser.parse_args())

def check_args(args):
    # --result_dir
    if os.path.isdir(os.path.join(args.result_dir, args.experiment_name)) and not args.resume:
        raise ValueError("experiment exist")

    os.makedirs(os.path.join(args.result_dir, args.experiment_name), exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, args.experiment_name, 'saved_models'), exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, args.experiment_name, 'images'), exist_ok=True)

    # --epoch
    try:
        assert args.n_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

def main():
    args = parse_args()

    unit = UNIT(args)
    unit.train()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()