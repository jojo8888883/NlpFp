import os
import time
import torch
import torch.nn as nn
import argparse
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image

# 创建小型数据集
class MiniVQADataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None, max_samples=500, max_classes=100):
        # 加载部分数据
        self.df = pd.read_csv(csv_path).sample(n=min(max_samples, 1000), random_state=42)
        self.transform = transform
        
        # 只保留部分类别
        self.answers = self.df['answers'].str.split('|').str[0].unique()[:max_classes]
        self.ans2idx = {ans: idx for idx, ans in enumerate(self.answers)}
        self.max_classes = max_classes
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 获取图片路径和问题
        image_path = self.df.iloc[idx]['image_path']
        question = self.df.iloc[idx]['question']
        question_id = self.df.iloc[idx]['question_id']
        
        # 读取图像
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            # 使用空白图像
            img = Image.new('RGB', (224, 224))
        
        # 对问题进行简单编码 (将字符串转为整数序列)
        question_tokens = question.split()[:20]  # 最大长度20
        question_ids = np.zeros(20, dtype=np.int64)
        for i, token in enumerate(question_tokens):
            question_ids[i] = hash(token) % 10000  # 简单的哈希
        
        # 获取答案标签
        answer = self.df.iloc[idx]['answers'].split('|')[0]
        # 映射到0~max_classes-1的类别
        if answer in self.ans2idx:
            label = self.ans2idx[answer]
        else:
            label = 0  # OOV答案
            
        # 应用转换
        if self.transform:
            img = self.transform(img)
            
        return {
            'image': img,
            'question': torch.tensor(question_ids),
            'answer': torch.tensor(label, dtype=torch.long),
            'question_id': question_id
        }

# 定义模型
class SimpleVQAModel(nn.Module):
    def __init__(self, num_classes=100, question_dim=20):
        super(SimpleVQAModel, self).__init__()
        # 使用预训练的MobileNetV3小型模型
        self.img_model = mobilenet_v3_small(pretrained=True)
        # 获取正确的特征维度
        img_feat_dim = 576  # MobileNetV3 Small的特征维度
        # 去掉分类器
        self.img_model.classifier = nn.Identity()
        
        # 简单问题编码器
        self.question_embedding = nn.Embedding(10000, 64)  # 使用10000大小的词汇表
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(img_feat_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, image, question):
        # 图像特征
        img_feat = self.img_model(image)
        
        # 问题特征
        q_embed = self.question_embedding(question)
        _, (hidden, _) = self.lstm(q_embed)
        q_feat = hidden.squeeze(0)
        
        # 融合
        combined = torch.cat([img_feat, q_feat], dim=1)
        output = self.fusion(combined)
        
        return output

# 训练函数
def train_minimal(args):
    # 创建保存目录
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 图像转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 创建小型数据集
    print("加载数据...")
    train_dataset = MiniVQADataset(
        args.train_csv, 
        transform=transform, 
        max_samples=args.max_samples,
        max_classes=args.max_classes
    )
    
    val_dataset = MiniVQADataset(
        args.val_csv, 
        transform=transform, 
        max_samples=args.max_samples//5,
        max_classes=args.max_classes
    )
    
    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"训练集样本: {len(train_dataset)}, 验证集样本: {len(val_dataset)}, 类别数: {args.max_classes}")
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleVQAModel(num_classes=args.max_classes).to(device)
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    print("开始训练...")
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 准备数据
            images = batch['image'].to(device)
            questions = batch['question'].to(device)
            labels = batch['answer'].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 显示进度
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} Batch {batch_idx+1}/{len(train_loader)} "
                      f"Loss: {train_loss/(batch_idx+1):.4f} "
                      f"Acc: {100.*train_correct/train_total:.2f}%")
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch['image'].to(device)
                questions = batch['question'].to(device)
                labels = batch['answer'].to(device)
                
                outputs = model(images, questions)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # 打印结果
        print(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {100.*train_correct/train_total:.2f}%, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {100.*val_correct/val_total:.2f}%")
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_minimal.pth'))
    print(f"模型保存至 {os.path.join(save_dir, 'model_minimal.pth')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='最小化VQA实验')
    
    # 数据参数
    parser.add_argument('--train_csv', type=str, default='data_proc/train.csv', 
                        help='训练CSV路径')
    parser.add_argument('--val_csv', type=str, default='data_proc/val.csv', 
                        help='验证CSV路径')
    parser.add_argument('--max_samples', type=int, default=500, 
                        help='最大样本数')
    parser.add_argument('--max_classes', type=int, default=100, 
                        help='最大类别数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='批量大小')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='训练轮数')
    parser.add_argument('--save_dir', type=str, default='runs/minimal_test', 
                        help='保存目录')
    
    args = parser.parse_args()
    train_minimal(args) 