import argparse
from solver import Solver

parser = argparse.ArgumentParser()

parser.add_argument('--mode', required=False, default='train', help='train or A or B')

# train
parser.add_argument('--batch_size', type=int, default=8, help='train batch size')
parser.add_argument('--dataset', required=False, default='horse2zebra', help='input dataset')

# test
parser.add_argument('--model_path', type=str, default='horse2zebra/models/generator_A_param.pkl')
parser.add_argument('--pic_path', type=str, default='datasets/horse2zebra/testB/n02391049_560.jpg')

# hyper-parameter
parser.add_argument('--decay_epoch', type=int, default=100, help='start decaying learning rate after this number')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lambdaA', type=float, default=10, help='lambdaA for cycle loss')
parser.add_argument('--lambdaB', type=float, default=10, help='lambdaB for cycle loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')

# model setup
parser.add_argument('--g_conv_dim', type=int, default=32)
parser.add_argument('--d_conv_dim', type=int, default=64)
parser.add_argument('--num_resnet', type=int, default=6, help='number of resnet blocks in generator')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
parser.add_argument('--flip', type=bool, default=True, help='random flip True of False')
parser.add_argument('--num_epochs', type=int, default=200, help='number of train epochs')

args = parser.parse_args()


def main():
    solver = Solver(args)
    if args.mode == 'train':
        solver.run()
    else:
        solver.sample(args.model_path, args.pic_path, args.mode)


if __name__ == '__main__':
    main()
