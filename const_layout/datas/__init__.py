from datas.rico import Rico
from datas.publaynet import PubLayNet
from datas.magazine import Magazine
from datas.infoppt import PPTNet


def get_dataset(name, split, transform=None):
    if name == 'rico':
        return Rico(split, transform)

    elif name == 'publaynet':
        return PubLayNet(split, transform)

    elif name == 'magazine':
        return Magazine(split, transform)

    elif name == 'infoppt':
        return PPTNet(split, transform)

    raise NotImplementedError(name)
