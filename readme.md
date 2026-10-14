## 运行项目

在项目目录下执行:
 `python train.py`
即可完成数据集下载与加载、训练、模型保存

训练完成以后,执行:
`python test.py`
即可随机选取一张图片进行测试


修改 **config.toml** 文件以更改配置


## bug 修改及优化
1. 训练损失改为累加 `loss.item()`, 避免带计算图的损失张量一直累积

2. 修复了保存模型前未更新 best_acc 的 bug。

3. 推理前调用 `model.eval()`, 并使用 `torch.inference_mode()`

4. 将按 epoch 显示的进度条更换为按 batch 显示

5. 修改了 acc 与 loss 计算策略 : 按样本数计算准确率和损失