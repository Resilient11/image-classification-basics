import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights
from tqdm import tqdm

from dataset import create_dataloaders
from utils import download_data, load_config, plot_loss_curves, set_seed, timer
from vit import ViT


def train_step(model, data_loader, loss_fn, optimizer: torch.optim.Optimizer, device):
    """完成一轮训练，返回按样本数统计的平均损失和准确率。"""
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for x, y in tqdm(data_loader, desc="训练", dynamic_ncols=True):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (pred.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / total_samples, total_correct / total_samples


def test_step(model, data_loader, loss_fn, device):
    """关闭梯度计算，返回一轮评估的平均损失和准确率。"""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.inference_mode():
        for x, y in tqdm(data_loader, desc="评估", dynamic_ncols=True):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (pred.argmax(dim=1) == y).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@timer
def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    save_dir: str | Path,
    device,
):
    """训练模型，记录每轮指标，并在评估准确率提高时保存权重。"""
    results = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    best_acc = 0
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        print(f"\n训练轮次：{epoch + 1}/{epochs}")
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_loader, loss_fn, device)

        print(
            f"训练准确率：{train_acc:.4f} 训练损失：{train_loss:.4f} "
            f"评估准确率：{test_acc:.4f} 评估损失：{test_loss:.4f}"
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_dir / "best.pt")

    return results


def build_model(model_config: dict, num_classes: int) -> ViT:
    """创建与 torchvision ViT-B/16 主干结构兼容的手写 ViT。"""
    required = {
        "img_size": 224,
        "in_channel": 3,
        "patch_size": 16,
        "embedding_dim": 768,
        "mlp_size": 3072,
        "num_transformer_layers": 12,
        "num_head": 12,
    }
    mismatches = {
        key: value for key, value in required.items()
        if model_config.get(key, value) != value
    }
    if mismatches:
        raise ValueError(
            f"加载 ViT-B/16 预训练权重需要以下配置：{mismatches}。"
            "改变这些结构参数后不能直接使用该预训练权重。"
        )
    return ViT(**model_config, num_classes=num_classes)


def main(config_path: str | Path | None = None):
    """读取配置，为手写 ViT 加载 ImageNet 主干权重并训练分类任务。"""
    config = load_config(config_path)
    training = config["train"]
    data = config["data"]
    selected_device = training.get("device", "auto")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if selected_device == "auto" else torch.device(selected_device)
    set_seed(training["seed"])

    image_path = download_data(
        source=data["source"],
        destination=data["dataset_name"],
        data_path=data["data_path"],
    )
    weights = ViT_B_16_Weights.DEFAULT
    image_transform = weights.transforms()
    train_loader, test_loader, class_names = create_dataloaders(
        train_dir=image_path / "train",
        test_dir=image_path / "test",
        transform=image_transform,
        batch_size=training["batch_size"],
        num_workers=training["num_workers"],
    )

    # 先创建手写模型，再将预训练主干的参数映射到对应层。
    model = build_model(config["model"], num_classes=len(class_names))
    model.load_pretrained_weights(weights.get_state_dict(progress=True, check_hash=True))
    if training["freeze_backbone"]:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier[1].parameters():
            parameter.requires_grad = True
    model = model.to(device)

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training["learning_rate"],
    )
    print(f"使用设备：{device}；模型：手写 ViT；权重目录：{training['save_dir']}")
    results = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        epochs=training["epochs"],
        save_dir=training["save_dir"],
        device=device,
    )
    plot_loss_curves(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用预训练权重训练手写 ViT")
    parser.add_argument("--config", type=Path, default=None, help="配置文件路径")
    main(parser.parse_args().config)
