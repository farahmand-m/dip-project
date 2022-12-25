def bind_subparser(subparsers):
    subparser = subparsers.add_parser('yolo')
    subparser.add_argument('--train-steps', type=int, default=None, help='maximum number of training steps')
    subparser.add_argument('--eval-steps', type=int, default=None, help='maximum number of evaluation steps')
    subparser.add_argument('--use-test', action='store_true', help='weather to use the "test" subset for eval')
    subparser.add_argument('--eval-only', action='store_true', help='only perform evaluation using stored weights')
    subparser.add_argument('--patch-size', nargs=3, type=int, default=(608, 608), help='size of extracted patches')
    subparser.add_argument('--anchor-sizes', nargs='+', type=int, default=(10, 20, 30), help='anchor sizes')
    subparser.add_argument('--anchor-ratios', nargs='+', type=float, default=(0.5, 1, 2), help='anchor ratios')
    subparser.add_argument('--lr', type=float, default=1e-3, help='adam optimizer learning rate')
    subparser.add_argument('--epochs', type=int, default=1, help='training epochs')
    subparser.add_argument('--batch-size', type=int, default=64, help='batch size')
    subparser.add_argument('--seed', type=int, default=7, help='random seed')
    return subparser
