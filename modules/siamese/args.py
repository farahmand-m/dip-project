def bind_subparser(subparsers):
    subparser = subparsers.add_parser('siamese')

    subparser.add_argument('--im-size', nargs=3, type=int, default=(224, 224), help='target image size')
    subparser.add_argument('--lr', type=float, default=1e-4, help='adam optimizer learning rate')
    subparser.add_argument('--epochs', type=int, default=1, help='training epochs')
    subparser.add_argument('--batch-size', type=int, default=64, help='batch size')
    subparser.add_argument('--seed', type=int, default=7, help='random seed')
    return subparser
