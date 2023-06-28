from datas.rico import Rico
from datas.publaynet import PubLayNet
from datas.magazine import Magazine


def get_dataset(name, split, json_dir=None, transform=None):
    if name == 'rico':
        return Rico(split, json_dir, transform)

    elif name == 'publaynet':
        return PubLayNet(split, transform)

    elif name == 'magazine':
        return Magazine(split, transform)

    raise NotImplementedError(name)
