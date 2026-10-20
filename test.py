import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights

from dataset import ImageFloderCustom
from train import build_model
from utils import download_image, load_config


def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str | Path,
    image_size: Tuple[int, int] = (224, 224),
    transform=None,
    device: str | torch.device = "cpu",
):
    """预测并绘制一张图片。

    参数：
        model：用于分类的模型，调用前需要加载训练后的权重。
        class_names：与训练时类别编号一致的名称列表。
        image_path：待预测图片路径。
        image_size：未指定 transform 时使用的缩放尺寸。
        transform：图片预处理；应与该模型训练时的预处理一致。
        device：运行预测的设备。

    手动缩放的默认分支保留给自定义模型；本项目入口会显式传入
    ViT 预训练权重所对应的预处理。
    """
    # 统一图片颜色模式，并在转换完成后关闭原文件。
    with Image.open(image_path) as source:
        img = source.convert("RGB")

    image_transform = transform if transform is not None else transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    model.to(device)
    model.eval()
    with torch.inference_mode():
        # 添加批次维度，使形状变为 [批次大小, 通道数, 高度, 宽度]。
        transformed_image = image_transform(img).unsqueeze(0)
        logits = model(transformed_image.to(device))
        probabilities = torch.softmax(logits, dim=1)
        label = probabilities.argmax(dim=1).item()
        probability = probabilities[0, label].item()

    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[label]} | Prob: {probability:.3f}")
    plt.axis(False)
    plt.show()


def main(config_path: str | Path | None = None):
    """加载配置和手写 ViT 的训练权重，对配置中的图片进行预测。"""
    config = load_config(config_path)
    training = config["train"]
    data = config["data"]
    selected_device = training.get("device", "auto")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if selected_device == "auto" else torch.device(selected_device)

    checkpoint_path = training["save_dir"] / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"没有找到权重：{checkpoint_path}，请先运行 train.py。")

    # 复用训练集的类别编号，不在预测时重新建立不同的映射。
    dataset_path = data["data_path"] / data["dataset_name"]
    class_names = ImageFloderCustom(dataset_path / "train").classes
    model = build_model(config["model"], num_classes=len(class_names))
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))

    image_path = download_image(
        source=config["test"]["image_url"],
        destination=config["test"]["image_path"],
    )
    print(f"使用设备：{device}；加载权重：{checkpoint_path}")
    pred_and_plot_image(
        model=model,
        class_names=class_names,
        image_path=image_path,
        transform=ViT_B_16_Weights.DEFAULT.transforms(),
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用训练后的手写 ViT 预测图片")
    parser.add_argument("--config", type=Path, default=None, help="配置文件路径")
    main(parser.parse_args().config)
