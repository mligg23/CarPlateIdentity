# 基于 PyTorch 和 OpenCV 的入门级车牌识别项目

这是中山大学智能工程学院大二的图像处理课程的第三次大作业。任务是处理车牌号码识别。

为了完成该作业，我参考了网上诸多资料，最终选择了如今的解决方案。即使用 OpenCV 中的传统图像方法对输入图像进行预处理，然后将车牌过滤和字符识别交给神经网络处理。

用到的两个神经网络都是由我自己使用 PyTorch 构建的，结构上借鉴了 LeNet5，十分简单，对于这种简单的任务可以说是绰绰有余，只需稍微训练即可取得不错的效果。

与其他的 repo 不同，这里我还给出了预训练好的网络模型 plate.pth 和 char.pth，便于大家复现出我的项目。

# 环境配置

Python 3.6

PyTorch 1.6.0+cu101

OpenCV 4.4.0

# 模型训练

需要注意的是，我的模型之前是在服务器上加载和运行的，模型结果是保存在第八张显卡上的。如果你的运行环境中显卡数目**少于八张**或者说第八张显卡的显存小于2G，则无法直接使用我的预训练模型。

如果需要自行训练模型，首先需要解压数据集 images.zip 使得其与 code 在同一目录下，然后还需要对 plateNeuralNet.py 和 charNeuralNet.py 进行一些修改，即

* 将文件头的 torch.cuda.set_device(7) 改为 torch.cuda.set_device(0)
* 将主函数中的 model = torch.load(train_model_path) 注释掉
* 恢复 model = char_cnn_net() 和 model = plate_cnn_net() 以及 model.apply(weights_init)

接下来只需要运行以下代码即可开始训练：

```bash
python3 plateNeuralNet.py
python3 charNeuralNet.py
```

由于训练需要一定时间，也可以用以下命令将进程挂到后台运行：

```bash
nohup python3 plateNeuralNet.py 1>plate.txt &
nohup python3 charNeuralNet.py 1>char.txt &
cat plate.txt
cat char.txt
```

使用 cat 命令即可实时地查看模型训练的进度。

# 项目运行

完成模型的训练后，替换掉原有的预训练模型。检查好数据所在的路径是否有误，然后就可以用以下命令开始运行项目：

```bash
python3 carPlateIdentity.py
```

如果需要测试特定图像，可以放进 images/test/ 目录下。