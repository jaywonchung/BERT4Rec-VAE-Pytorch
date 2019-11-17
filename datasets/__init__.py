from .ml_1m import ML1MDataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
