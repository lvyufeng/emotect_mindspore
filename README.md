# 对话情绪识别(Mindspore Version)

## 模型简介

对话情绪识别（Emotion Detection，简称EmoTect），专注于识别智能对话场景中用户的情绪，针对智能对话场景中的用户文本，自动判断该文本的情绪类别并给出相应的置信度，情绪类型分为积极、消极、中性。
<!-- 
效果上，我们基于百度自建测试集（包含闲聊、客服）和 nlpcc2014 微博情绪数据集，进行评测，效果如下表所示，此外我们还开源了百度基于海量数据训练好的模型，该模型在聊天对话语料上 Finetune 之后，可以得到更好的效果。

| 模型 | 闲聊 | 客服 | 微博 |
| :------| :------ | :------ | :------ |
| BOW | 90.2% | 87.6% | 74.2% |
| LSTM | 91.4% | 90.1% | 73.8% |
| Bi-LSTM | 91.2%  | 89.9%  | 73.6% |
| CNN | 90.8% |  90.7% | 76.3%  |
| TextCNN |  91.1% | 91.0% | 76.8% |
| BERT | 93.6% | 92.3%  | 78.6%  |
| ERNIE | 94.4% | 94.0% | 80.6% |
-->

## 快速开始

### 环境依赖

1. Python 3 版本: 3.7.5
2. PaddlePadlle: 2.0.0
3. MindSpore: 1.1.1 (GPU或Ascend版本)

- 环境安装:

   推荐使用conda构建虚拟环境:
   ```bash
   conda create -n mindspore python=3.7.5 cudatoolkit=10.1 cudnn=7.6
   ```
   安装MindSpore
   ```bash
   # if using GPU
   pip install mindspore-gpu==1.1.1
   # if using Ascend
   pip install mindspore-Ascend==1.1.1
   ```
   安装其他依赖库
   ```bash
   pip install -r requirement.txt
   ```

### 数据准备

#### **百度公开数据集**

这里我们使用百度提供的一份已标注的、经过分词预处理的机器人聊天数据集，运行数据下载脚本:
```bash
sh script/download_data.sh
```
运行成功后，会生成文件夹 ```data```，其目录结构如下：

```text
.
├── train.tsv       # 训练集
├── dev.tsv         # 验证集
├── test.tsv        # 测试集
├── infer.tsv       # 待预测数据
├── vocab.txt       # 词典

```
运行数据格式转换脚本, 将数据集转为MindRecord格式:
```bash
sh scripts/convert_dataset.sh
```

#### **自定义数据**

数据由两列组成，以制表符（'\t'）分隔，第一列是情绪分类的类别（0表示消极；1表示中性；2表示积极），第二列是以空格分词的中文文本，如下示例，文件为 utf8 编码。

```text
label   text_a
0   谁 骂人 了 ？ 我 从来 不 骂人 ， 我 骂 的 都 不是 人 ， 你 是 人 吗 ？
1   我 有事 等会儿 就 回来 和 你 聊
2   我 见到 你 很高兴 谢谢 你 帮 我
```

注：项目提供了分词预处理脚本（src/tokenizer.py），可供用户使用.

### 预训练模型权重迁移

EmoTect基于海量数据训练好的对话情绪识别模型（基于TextCNN、ERNIE等模型训练），可供用户直接使用，可通过以下方式下载。

```shell
sh script/download_model.sh
```

以上两种方式会将预训练的ERNIE模型，保存在```pretrain_models```目录下，可直接运行:
```bash
sh scripts/paddle_to_mindspore.sh
```
将Paddle存储的预训练模型参数权重迁移至MindSpore, 进行后续的微调、评估、预测。

#### 迁移模型评估
模型参数迁移后可直接对其进行评估：
```bash
sh scripts/run_classifier_eval.sh
# 结果示例：
# ==============================================================
# acc_num 979 , total_num 1036, accuracy 0.944981
# ==============================================================
```

### 基于 ERNIE 进行 Finetune

ERNIE 是百度自研的基于海量数据和先验知识训练的通用文本语义表示模型，基于 ERNIE 进行 Finetune，能够提升对话情绪识别的效果。

#### 模型训练

需要先下载 ERNIE 模型，使用如下命令：

```shell
mkdir -p pretrain_models/ernie
cd pretrain_models/ernie
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz -O ERNIE_stable-1.0.1.tar.gz
tar -zxvf ERNIE_stable-1.0.1.tar.gz
```

然后修改```script/paddle_to_midnspore.sh``` 脚本中 ```MODEL_PATH``` 参数为ERNIE模型目录，再执行命令：

```shell
#--input_dir ./pretrain_models/ernie
sh script/paddle_to_midnspore.sh
```

将ERNIE迁移至Mindspore后，执行训练脚本:
```bash
sh scripts/run_classifier_finetune.sh
```
默认使用GPU进行训练，模型保存在 ```./save_models/ernie_finetune.ckpt```。

#### 模型评估

根据训练结果，可选择最优的step进行评估，修改```scripts/run_classifier_eval.sh``` 脚本中```load_finetune_checkpoint_path``` 参数，然后执行

```shell
#--load_finetune_checkpoint_path="${SAVE_PATH}/ernie_finetune.ckpt"
sh scripts/run_classifier_eval.sh
```
<!-- 
#### 模型推断

修改```run_ernie.sh``` 脚本中 infer 函数 ```init_checkpoint``` 参数，然后执行

```shell
#--init_checkpoint./save/step_907
sh run_ernie.sh infer

'''
# 输出结果示例
Load model from ./save_models/ernie/step_907
Final test result:
1      0.000803      0.998870      0.000326
0      0.976585      0.021535      0.001880
1      0.000572      0.999153      0.000275
1      0.001113      0.998502      0.000385
'''
``` -->