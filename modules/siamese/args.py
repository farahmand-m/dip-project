def bind_subparser(subparsers):
    subparser = subparsers.add_parser('siamese')
    subparser.add_argument('--train-steps', type=int, default=None, help='maximum number of training steps')
    subparser.add_argument('--eval-steps', type=int, default=None, help='maximum number of evaluation steps')
    subparser.add_argument('--use-test', action='store_true', help='weather to use the "test" subset for eval')
    subparser.add_argument('--eval-only', action='store_true', help='only perform evaluation using stored weights')
    subparser.add_argument('--im-size', nargs=3, type=int, default=(224, 224), help='target image size')
    subparser.add_argument('--lr', type=float, default=1e-4, help='adam optimizer learning rate')
    subparser.add_argument('--epochs', type=int, default=1, help='training epochs')
    subparser.add_argument('--batch-size', type=int, default=64, help='batch size')
    subparser.add_argument('--seed', type=int, default=7, help='random seed')
    return subparser
