# dataset.py
import os
from utils import timer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


@timer
def load_data(config : dict):
    """
    加载训练集和测试集。

    Returns:
        train_loader: 训练集 DataLoader
        test_loader: 测试集 DataLoader
        class_name: 类别名称列表
    """

    data_path = config['data']['data_path']
    os.makedirs(data_path, exist_ok=True)
    train_data = datasets.FashionMNIST(root=data_path, train=True, transform= transforms.ToTensor(), download=True)
    test_data  = datasets.FashionMNIST(root=data_path, train=False, transform= transforms.ToTensor(), download=True)

    class_name = train_data.classes


    train_loader = DataLoader(train_data, 
                            batch_size=config['train']['batch_size'], 
                            shuffle=True, 
                            num_workers=config['train']['num_workers'])

    test_loader = DataLoader(test_data,
                            batch_size=config['train']['batch_size'],
                            num_workers=config['train']['num_workers'])
    
    return train_loader, test_loader, class_name



if __name__ == "__main__":
    import tomllib

    with open('config.toml', 'rb') as f:
        config = tomllib.load(f)

    load_data(config)
