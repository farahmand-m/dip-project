import json


def load_dataset(data_dir):
    filenames = ('train.json', 'valid.json', 'test.json')
    for filename in filenames:
        with open(data_dir / 'annotations' / filename) as stream:
            yield json.load(stream)
