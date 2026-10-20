import os
import stat
import tempfile
import time
import tomllib
import zipfile

from functools import wraps
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def load_config(path=None):
    """读取配置，并以配置文件所在目录为基准解析数据、权重和预测图片路径。"""
    config_path = Path(path).expanduser().resolve() if path else PROJECT_ROOT / "config.toml"
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    defaults = (
        ("data", "data_path", "./data"),
        ("train", "save_dir", "./runs/handwritten_pretrained_vit"),
        ("test", "image_path", "./data/pizza_steak_sushi/04-pizza-dad.jpeg"),
    )
    for section, key, default in defaults:
        values = config.setdefault(section, {})
        configured_path = Path(values.get(key, default)).expanduser()
        if not configured_path.is_absolute():
            configured_path = config_path.parent / configured_path
        values[key] = configured_path.resolve()
    config["data"].setdefault("dataset_name", "pizza_steak_sushi")
    return config


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} 耗时: {end - start:.2f} 秒")
        return result
    return wrapper


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def plot_loss_curves(results):
    """绘制训练结果字典中的损失与准确率曲线。

    参数：
        results：包含 train_loss、train_acc、test_loss、test_acc 列表的字典。
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # 绘制损失曲线。
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # 绘制准确率曲线。
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


def _is_valid_image(path):
    """同时检查图片容器和像素内容，识别空文件、损坏文件与截断图片。"""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def _validate_dataset(path):
    """检查图片完整性；测试类别可以是训练类别的非空子集。"""
    class_sets = []
    for split in ("train", "test"):
        split_path = path / split
        if not split_path.is_dir():
            return False
        classes = {folder.name for folder in split_path.iterdir() if folder.is_dir()}
        if not classes:
            return False
        class_sets.append(classes)
        for class_name in classes:
            images = [
                file for file in (split_path / class_name).iterdir()
                if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES
            ]
            if not images or not all(_is_valid_image(image) for image in images):
                return False
    return class_sets[1] <= class_sets[0]


def _download_file(source, destination):
    """流式下载到调用方提供的临时文件，网络异常交由调用方处理。"""
    with requests.get(source, stream=True, timeout=(10, 60)) as response:
        response.raise_for_status()
        with destination.open("wb") as target:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    target.write(chunk)


def _extract_zip_safely(archive, destination):
    """验证压缩包并拒绝越界路径与符号链接，再解压到临时目录。"""
    with zipfile.ZipFile(archive) as zip_file:
        members = zip_file.infolist()
        for member in members:
            name = member.filename.replace("\\", "/")
            relative_path = PurePosixPath(name)
            mode = member.external_attr >> 16
            if (relative_path.is_absolute() or ".." in relative_path.parts
                    or any(":" in part for part in relative_path.parts)
                    or stat.S_ISLNK(mode)):
                raise ValueError(f"压缩包包含不安全的路径：{member.filename}")
            target = (destination / Path(*relative_path.parts)).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"压缩包路径超出目标目录：{member.filename}")
        damaged_file = zip_file.testzip()
        if damaged_file:
            raise zipfile.BadZipFile(f"压缩包文件校验失败：{damaged_file}")
        zip_file.extractall(destination)


def _publish_dataset(source, destination):
    """逐个合并已验证的文件，保留已有有效图片和用户额外放置的文件。"""
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    for file in source.rglob("*"):
        if not file.is_file():
            continue
        target = destination / file.relative_to(source)
        if not target.resolve().is_relative_to(destination_root):
            raise ValueError(f"目标目录包含指向外部的路径：{target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            if target.suffix.lower() not in IMAGE_SUFFIXES or _is_valid_image(target):
                continue
        os.replace(file, target)


@timer
def download_data(source: str,
                  destination: str,
                  remove_source: bool = True,
                  data_path=PROJECT_ROOT / "data") -> Path:
    """可靠下载并解压包含 train/test 类别目录的图片数据集。

    参数：
        source：数据集 ZIP 文件地址。
        destination：数据根目录下的数据集目录名。
        remove_source：是否在解压后移除下载的 ZIP；为 False 时保留到数据根目录。
        data_path：数据根目录，默认使用本项目的 data 目录。

    只有类别和图片检查全部通过才跳过下载。下载和解压先在临时目录完成，
    成功后合并文件；失败后再次调用可以继续修复，不会将空目录误认为完整数据。
    """
    data_path = Path(data_path).expanduser().resolve()
    image_path = data_path / destination
    train_path, test_path = image_path / "train", image_path / "test"
    if train_path.is_dir() and test_path.is_dir():
        train_classes = {folder.name for folder in train_path.iterdir() if folder.is_dir()}
        test_classes = {folder.name for folder in test_path.iterdir() if folder.is_dir()}
        unknown_classes = test_classes - train_classes
        if train_classes and test_classes and unknown_classes:
            raise ValueError(
                f"测试集包含训练集中不存在的类别：{sorted(unknown_classes)}。"
                "请先检查数据划分，自动下载不会改变现有类别映射。"
            )
    if _validate_dataset(image_path):
        print(f"[INFO] 数据集已通过完整性检查：{image_path}")
        return image_path

    data_path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dataset-", dir=data_path) as temp_dir:
        temp_path = Path(temp_dir)
        archive = temp_path / "dataset.zip"
        extracted = temp_path / "extracted"
        extracted.mkdir()
        print(f"[INFO] 正在下载数据集：{source}")
        _download_file(source, archive)
        _extract_zip_safely(archive, extracted)

        # 兼容 ZIP 直接包含 train/test，或外面再包一层数据集目录的情况。
        dataset_source = extracted
        if not _validate_dataset(dataset_source):
            dataset_source = extracted / Path(destination).name
        if not _validate_dataset(dataset_source):
            raise ValueError(
                "下载的数据集不完整：需要图片可读、类别非空，且测试类别不超出训练类别范围的 train/test 目录。"
            )

        _publish_dataset(dataset_source, image_path)
        if not _validate_dataset(image_path):
            raise ValueError(
                f"数据集合并后仍未通过检查：{image_path}。"
                "请检查额外类别目录或损坏的自定义图片；已有用户文件已保留。"
            )
        if not remove_source:
            archive_name = PurePosixPath(urlparse(source).path).name or "dataset.zip"
            os.replace(archive, data_path / archive_name)
    return image_path


def download_image(source, destination):
    """下载单张图片；先验证临时文件，再替换损坏文件，已有有效图片直接复用。"""
    destination = Path(destination).expanduser().resolve()
    if _is_valid_image(destination):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".image-", dir=destination.parent) as temp_dir:
        downloaded = Path(temp_dir) / "image"
        _download_file(source, downloaded)
        if not _is_valid_image(downloaded):
            raise ValueError(f"下载内容不是完整有效的图片：{source}")
        os.replace(downloaded, destination)
    return destination


if __name__ == "__main__":
    image_path = download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
        destination="pizza_steak_sushi",
    )
    print(image_path)
