import numpy as np
import os
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

class VQADataset(Dataset):
    """
    视觉问答数据集类，支持从CSV文件加载数据
    """
    def __init__(self, csv_path, question_vocab_path, answer_vocab_path, max_qu_len=30, transform=None):
        # 加载CSV数据
        self.data = pd.read_csv(csv_path)
        
        # 加载词汇表
        self.qu_vocab = Vocab(question_vocab_path)
        self.ans_vocab = Vocab(answer_vocab_path)
        
        self.max_qu_len = max_qu_len
        self.transform = transform
        self.labeled = 'answers' in self.data.columns

    def __getitem__(self, idx):
        # 获取图片路径和问题
        image_path = self.data.iloc[idx]['image_path']
        question_text = self.data.iloc[idx]['question']
        question_id = self.data.iloc[idx]['question_id']
        
        # 读取并处理图像
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"加载图片出错: {image_path}, 错误: {e}")
            # 返回一个黑色图像作为替代
            img = Image.new('RGB', (224, 224))
        
        # 分词并转换为索引
        qu_tokens = tokenizer(question_text)
        qu2idx = np.array([self.qu_vocab.word2idx('<pad>')] * self.max_qu_len)
        qu2idx[:min(len(qu_tokens), self.max_qu_len)] = [
            self.qu_vocab.word2idx(token) for token in qu_tokens[:self.max_qu_len]
        ]
        
        # 准备样本字典
        sample = {
            'image': img, 
            'question': qu2idx, 
            'question_id': question_id
        }

        # 如果有标签（训练/验证集），处理答案
        if self.labeled:
            # 解析管道分隔的答案列表
            answers = self.data.iloc[idx]['answers'].split('|')
            if len(answers) > 0:
                # 转换为索引，并随机选择一个有效答案
                ans2idx = [self.ans_vocab.word2idx(ans) for ans in answers if ans.strip()]
                if ans2idx:
                    sample['answer'] = np.random.choice(ans2idx)
                else:
                    sample['answer'] = self.ans_vocab.word2idx('<unk>')
            else:
                sample['answer'] = self.ans_vocab.word2idx('<unk>')

        # 应用图像转换
        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

    def __len__(self):
        return len(self.data)

def tokenizer(sentence):
    """简单的分词器"""
    import re
    regex = re.compile(r'(\W+)')
    tokens = regex.split(sentence.lower())
    tokens = [w.strip() for w in tokens if len(w.strip()) > 0]
    return tokens

def get_data_loader(train_csv, val_csv, question_vocab_path, answer_vocab_path, 
                   batch_size=128, max_qu_len=30, num_workers=4):
    """
    创建训练和验证数据加载器
    """
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 创建数据集
    train_dataset = VQADataset(
        csv_path=train_csv,
        question_vocab_path=question_vocab_path,
        answer_vocab_path=answer_vocab_path,
        max_qu_len=max_qu_len,
        transform=transform
    )
    
    val_dataset = VQADataset(
        csv_path=val_csv,
        question_vocab_path=question_vocab_path,
        answer_vocab_path=answer_vocab_path,
        max_qu_len=max_qu_len,
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return {'train': train_loader, 'val': val_loader}

class Vocab:
    """
    词汇表类，负责词到索引的转换
    """
    def __init__(self, vocab_file):
        self.vocab = self.load_vocab(vocab_file)
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def load_vocab(self, vocab_file):
        with open(vocab_file) as f:
            vocab = [v.strip() for v in f]
        return vocab

    def word2idx(self, vocab):
        if vocab in self.vocab2idx:
            return self.vocab2idx[vocab]
        else:
            return self.vocab2idx['<unk>']

    def idx2word(self, idx):
        if 0 <= idx < len(self.vocab):
            return self.vocab[idx]
        return '<unk>'
