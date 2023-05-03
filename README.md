# CRNN_ctc_digital_recognize
### 基于卷积循环神经网络的数字识别

> Author: github.com/SummerColdWind
> 
> Dataset provider: github.com/TsangHans
---

## 1.准备数据集
> 数据集中图片应为png格式并统一大小
> 
> 数据集中标注文件格式为“图片文件名\t图片文字”，例如:

```
30.png	-29
31.png	-30
32.png	-29
33.png	-27
34.png	-28
35.png	-27
```

## 2.划分训练集与测试集
> 建议使用 ./tools 下 **split_train_and_test_dataset.py** 来执行划分操作

## 3.修改配置文件
> 配置文件默认在 ./configs/global. yml
```
# 配置文件说明
Global:
  use_gpu: 是否使用gpu
  epoch_num: 训练总epoch数
  save_model_dir: 模型保存文件夹，默认为./output
  save_epoch_step: 每几次epoch保存一次模型
  learning_rate: 学习率，默认为0.001
  character_dict_path: 字典，默认在./configs/dict.txt
  max_text_length: 最大文字长度

Train:
  data_dir: 训练集文件夹路径
  label_file_dir: 训练集标注文件路径
  shuffle: 是否打乱，默认为True
  batch_size: 训练批次大小

Test:
  data_dir: 测试集文件夹路径
  label_file_dir: 测试集标注文件路径
  shuffle: 是否打乱，False
  batch_size: 测试批次大小
```

## 4.启动训练
> 运行项目内 train.py

## 5.进行推理
> 运行项目内 infer.py
```python
# 选择训练好的模型路径
model_path = './output/best.pth'
# 选择要推理的图片路径
image_path = './example.png'
```