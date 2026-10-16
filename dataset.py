# dataset.py
import os

from PIL import Image
from pathlib import Path
from typing import Any, Tuple, List, Dict
from utils import timer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


@timer
def load_data(config : dict):
    """
    加载训练集和测试集 : FashionMNIST 

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



class ImageFloderCustom(Dataset):
    def __init__(self, target_dir, transforms=None) -> None:
        super().__init__()
        self.paths = list(Path(target_dir).glob('*/*.jpg'))
        self.transform = transforms
        self.classes , self.class_to_idx = self.find_classes(target_dir)


    def find_classes(self, directory:Path) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(x.name for x in os.scandir(directory) if x.is_dir())
        if not classes:
            raise FileNotFoundError(f'Could not find any classes in {directory}')

        class_to_idx = {class_name : i for i, class_name in enumerate(classes)}

        return classes, class_to_idx

    def load_img(self, index:int) -> Image.Image:
        img_path = self.paths[index]
        return Image.open(img_path)


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) -> Any:
        img = self.load_img(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx




if __name__ == "__main__":
    import tomllib

    with open("config.toml", 'rb') as f:
        config = tomllib.load(f)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_path = config['data']['data_path_custom']+'/train'
    test_path  = config['data']['data_path_custom']+'/test'

    train_data = ImageFloderCustom(train_path, transforms=transform)
    test_data =  ImageFloderCustom(test_path , transforms=transform)

    train_dataloader = DataLoader(
        train_data,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    img_custom, label_custom = next(iter(train_dataloader))

    # Batch size will now be 1, try changing the batch_size parameter above and see what happens
    print(f"Image shape: {img_custom.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label_custom.shape}")