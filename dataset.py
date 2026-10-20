# 图片数据集与数据加载器。
import os

from PIL import Image
from pathlib import Path
from typing import Any, Tuple, List, Dict
from utils import timer
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset



class ImageFloderCustom(Dataset):
    def __init__(self, target_dir : str|Path, transforms=None, class_to_idx=None) -> None:
        super().__init__()
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
        self.paths = sorted(
            path for path in Path(target_dir).glob("*/*")
            if path.is_file() and path.suffix.lower() in image_extensions
        )
        self.transform = transforms
        found_classes, found_mapping = self.find_classes(target_dir)
        if class_to_idx is None:
            self.classes, self.class_to_idx = found_classes, found_mapping
        else:
            # 测试集沿用训练集编号；允许缺类，但新增类别必须主动报错。
            self.class_to_idx = dict(class_to_idx)
            if sorted(self.class_to_idx.values()) != list(range(len(self.class_to_idx))):
                raise ValueError("类别编号必须从零开始，连续且不重复。")
            unknown_classes = sorted(set(found_classes) - set(self.class_to_idx))
            if unknown_classes:
                raise ValueError(f"测试集包含训练集中不存在的类别：{unknown_classes}。")
            self.classes = sorted(self.class_to_idx, key=self.class_to_idx.get)


    def find_classes(self, directory:str|Path) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(x.name for x in os.scandir(directory) if x.is_dir())
        if not classes:
            raise FileNotFoundError(f'在 {directory} 中未找到类别文件夹。')

        class_to_idx = {class_name : i for i, class_name in enumerate(classes)}

        return classes, class_to_idx

    def load_img(self, index:int) -> Image.Image:
        img_path = self.paths[index]
        # 统一灰度图等颜色模式，并及时关闭原文件。
        with Image.open(img_path) as source:
            return source.convert("RGB")


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

@timer
def create_dataloaders(train_dir : str|Path,
              test_dir : str|Path,
              transform :  transforms.Compose,
              batch_size : int,
              num_workers : int):
    """
    加载训练集和测试集

    返回训练集加载器、测试集加载器与训练集类别名称列表。
    """

    train_data = ImageFloderCustom(train_dir, transforms= transform)
    test_data = ImageFloderCustom(
        test_dir, transforms=transform, class_to_idx=train_data.class_to_idx
    )

    class_name = train_data.classes


    train_loader = DataLoader(train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)

    test_loader = DataLoader(test_data,
                            batch_size=batch_size,
                            num_workers=num_workers)

    return train_loader, test_loader, class_name








if __name__ == "__main__":
    from utils import load_config

    config = load_config()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset_path = Path(config["data"]["data_path"]) / config["data"]["dataset_name"]
    train_path = dataset_path / "train"
    test_path = dataset_path / "test"

    _, _, class_name = create_dataloaders(
        train_path, test_path, transform,
        config["train"]["batch_size"], config["train"]["num_workers"],
    )

    print(class_name)




    # train_data = ImageFloderCustom(train_path, transforms=transform)
    # test_data =  ImageFloderCustom(test_path , transforms=transform)

    # train_dataloader = DataLoader(
    #     train_data,
    #     batch_size=config['train']['batch_size'],
    #     shuffle=True,
    #     num_workers=0
    # )
    # test_dataloader = DataLoader(
    #     test_data,
    #     batch_size=config['train']['batch_size'],
    #     shuffle=False,
    #     num_workers=0
    # )

    # img_custom, label_custom = next(iter(train_dataloader))

    # 修改上面的 batch_size，可观察图片与标签批次形状的变化。
    # print(f"图片形状：{img_custom.shape} -> [批大小, 颜色通道数, 高度, 宽度]")
    # print(f"标签形状：{label_custom.shape}")
