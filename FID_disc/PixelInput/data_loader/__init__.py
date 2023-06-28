from data_loader.datasets import FlatDirectoryImageDataset, FoldersDistributedDataset, FlatDirectoryNumpyDataset
from data_loader.transforms import get_transform


def make_dataset(dir, is_folder=False, resolution=1024, is_img=True):
    if is_img:
        if is_folder:
            Dataset = FoldersDistributedDataset
        else:
            Dataset = FlatDirectoryImageDataset
        transform = get_transform(new_size=(resolution, resolution))
    else:
        Dataset = FlatDirectoryNumpyDataset
        transform = None

    _dataset = Dataset(data_dir=dir, transform=transform)
    return _dataset


def get_data_loader(dataset, batch_size, num_workers):
    """
    generate the data_loader from the given dataset
    :param dataset: dataset for training (Should be a PyTorch dataset)
                    Make sure every item is an Image
    :param batch_size: batch size of the data
    :param num_workers: num of parallel readers
    :return: dl => data_loader for the dataset
    """
    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=False
    )

    return dl