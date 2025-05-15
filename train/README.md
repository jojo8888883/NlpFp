## Visual Question Answering (VQA) 模型

本项目实现了一个基于CNN+LSTM的视觉问答(VQA)模型，并通过多种技术手段进行了改进，包括VGG19微调、双向LSTM和特征级联融合。

### 性能对比

下表展示了我们改进模型与原始论文的性能对比：

| 模型 | All | Yes/No | Number | Other |
| ------ | ------ | ------ | ------ | ------ |
| 基线模型 | 46.23% | 67.5% | 30.83% | 34.12% |
| 改进模型（VGG微调+双向LSTM+特征级联+Focal Loss） | 54.22% | 73.46% | 35.18% | 41.38% |

### 数据集

VQA v2.0
- 82,783 MS COCO训练图像，40,504 MS COCO验证图像和81,434 MS COCO测试图像 
- 443,757个训练问题，214,354个验证问题和447,793个测试问题
- 4,437,570个训练答案和2,143,540个验证答案（每个问题10个）

### 模型改进

- **图像编码器**：使用预训练的VGG19，微调最后两个卷积块（conv5），提取1024维特征
- **问题编码器**：使用双向LSTM处理问题，更好地捕捉上下文信息
- **特征融合**：使用特征级联而非元素乘法，通过全连接层和GELU激活融合两种模态
- **损失函数**：支持使用Focal Loss处理类别不平衡问题
- **优化策略**：
  - 参数分组（CNN主干使用较小学习率）
  - 随机权重平均(SWA)提高泛化能力
  - 早停机制防止过拟合

### 使用方法

#### 1. 数据预处理
```bash
# 调整图像大小到统一尺寸（224x224）
python preprocess/resize_images.py \
  --image_root data_raw/train2014 data_raw/val2014 \
  --save_root data_proc/images \
  --size 224

# 生成问题词表
python preprocess/make_vocab.py \
  --questions_json data_raw/v2_OpenEnded_mscoco_train2014_questions.json \
  --min_freq 1 \
  --out_txt question_vocabs.txt

# 生成CSV数据集
python preprocess/make_dataset_csv.py \
  --train_questions data_raw/v2_OpenEnded_mscoco_train2014_questions.json \
  --val_questions data_raw/v2_OpenEnded_mscoco_val2014_questions.json \
  --train_annotations data_raw/v2_mscoco_train2014_annotations.json \
  --val_annotations data_raw/v2_mscoco_val2014_annotations.json \
  --images_dir data_proc/images \
  --output_dir data_proc
```

#### 2. 模型训练
```bash
python model/train.py \
  --train_csv data_proc/train.csv \
  --val_csv data_proc/val.csv \
  --question_vocab data_proc/question_vocabs.txt \
  --answer_vocab data_proc/annotation_vocabs.txt \
  --lr 2e-4 \
  --cnn_lr 2e-5 \
  --batch_size 128 \
  --epochs 12 \
  --focal_loss \
  --use_swa \
  --save_dir runs/vgg_ft_bilstm_concat
```

#### 3. 模型测试与评测
```bash
# 在验证集上进行推理
python model/test.py \
  --val_csv data_proc/val.csv \
  --question_vocab data_proc/question_vocabs.txt \
  --answer_vocab data_proc/annotation_vocabs.txt \
  --ckpt runs/vgg_ft_bilstm_concat/best_acc.pth \
  --out_json runs/vgg_ft_bilstm_concat/pred.json

# 使用官方工具评测
python VQA_evaluation/PythonEvaluationTools/vqaEvalDemo.py \
  --annFile data_raw/v2_mscoco_val2014_annotations.json \
  --quesFile data_raw/v2_OpenEnded_mscoco_val2014_questions.json \
  --resFile runs/vgg_ft_bilstm_concat/pred.json
```

#### 4. 可视化与消融实验
```bash
# Grad-CAM可视化
python model/visualize.py \
  --image data_raw/test_images/001.jpg \
  --question "这个人在干什么?" \
  --ckpt runs/vgg_ft_bilstm_concat/best_acc.pth \
  --output visualization/example_001.png

# 消融实验
python model/ablation.py \
  --train_csv data_proc/train.csv \
  --val_csv data_proc/val.csv \
  --question_vocab data_proc/question_vocabs.txt \
  --answer_vocab data_proc/annotation_vocabs.txt \
  --result_dir ablation_results
```

### 文件结构

- `preprocess/` - 数据预处理脚本
  - `resize_images.py` - 调整图像大小
  - `make_vocab.py` - 生成词汇表
  - `make_dataset_csv.py` - 生成CSV数据集
  
- `model/` - 模型实现
  - `model.py` - 模型架构定义
  - `build_dataset.py` - 数据集加载
  - `train.py` - 训练脚本
  - `test.py` - 测试与推理脚本
  - `visualize.py` - Grad-CAM可视化
  - `ablation.py` - 消融实验

### 环境要求

- Python 3.7+
- PyTorch 1.7+
- torchvision
- pandas
- matplotlib
- pillow
- tqdm
- transformers
