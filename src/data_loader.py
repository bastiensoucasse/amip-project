from torch.utils.data import DataLoader

from datasets import load_dataset

DATASETS: dict[str, str] = {
    'set5': 'eugenesiow/Set5',
    'set14': 'eugenesiow/Set14',
    'bsd100': 'eugenesiow/BSD100'
}


def load(dataset: str, batch_size: int) -> DataLoader:
    '''Loads a dataset from its id ('set5', 'set14' or 'bsd100').'''

    if dataset not in DATASETS:
        raise ValueError(f'Invalid Dataset: \"{dataset}\".')

    return DataLoader(load_dataset(DATASETS[dataset]), batch_size=batch_size, shuffle=True, num_workers=2)
