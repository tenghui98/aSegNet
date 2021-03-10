import argparse
def parser():
    parser = argparse.ArgumentParser(description='Deeplab training')
    parser.add_argument('--loss-type', type=str, default='l1ce',
                        choices=['ce', 'focal','dice'],
                        help='loss func type (default: ce)')
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch_size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use_balanced_weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--cuda', action='store_false', default=
                        True, help='enables CUDA training')
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    parser.add_argument('--category', type=str, default='baseline',
                        help='category of cdw2014 dataset')
    parser.add_argument('--scene', type=str, default='highway',
                        help='scene in category')
    parser.add_argument('--train_rate', type=int, default=0.8,
                        help='splitting rate')
    parser.add_argument('--motion', action='store_false', default=True,
                        help='label 170 is set to 0 or 2')
    parser.add_argument('--cbam', action='store_true', default=False,
                        help='cbam attention')
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 100

    if args.batch_size is None:
        args.batch_size = 1

    if args.lr is None:
        args.lr = 0.001

    if args.checkname is None:
        # args.checkname = 'deeplab-cdw2014'
        args.checkname = 'deepfeg-cdw2014'
    return args