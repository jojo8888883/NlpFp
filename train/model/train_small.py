import os
import time
import torch
import torch.nn as nn
import argparse
import random
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from model import VQAModel, FocalLoss
from build_dataset import VQADataset
from torchvision import transforms

def train_small_sample(args):
    """
    训练VQA模型的小规模实验版本
    """
    # 创建保存目录
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    
    # 记录配置
    with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 加载数据
    print("正在加载数据...")
    # 加载小规模数据
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    
    # 随机选择一小部分数据
    train_sample = train_df.sample(n=min(args.sample_size, len(train_df)), random_state=42)
    val_sample = val_df.sample(n=min(args.sample_size//5, len(val_df)), random_state=42)
    
    # 保存采样后的CSV
    train_sample_path = os.path.join(save_dir, 'train_sample.csv')
    val_sample_path = os.path.join(save_dir, 'val_sample.csv')
    train_sample.to_csv(train_sample_path, index=False)
    val_sample.to_csv(val_sample_path, index=False)
    
    # 创建数据集
    train_dataset = VQADataset(
        csv_path=train_sample_path,
        question_vocab_path=args.question_vocab,
        answer_vocab_path=args.answer_vocab,
        max_qu_len=args.max_qu_len,
        transform=transform
    )
    
    val_dataset = VQADataset(
        csv_path=val_sample_path,
        question_vocab_path=args.question_vocab,
        answer_vocab_path=args.answer_vocab,
        max_qu_len=args.max_qu_len,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 获取词汇表大小
    qu_vocab_size = train_dataset.qu_vocab.vocab_size
    ans_vocab_size = train_dataset.ans_vocab.vocab_size
    print(f"问题词汇表大小: {qu_vocab_size}, 答案词汇表大小: {ans_vocab_size}")
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VQAModel(
        feature_size=args.feature_size,
        qu_vocab_size=qu_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed=args.word_embed,
        hidden_size=args.hidden_size,
        num_hidden=args.num_hidden
    ).to(device)
    
    # 参数分组 - CNN主干使用较小学习率
    backbone_params = list(model.img_encoder.model.parameters())
    rest_params = [p for p in model.parameters() if p not in set(backbone_params)]
    
    optimizer = AdamW([
        {'params': backbone_params, 'lr': args.cnn_lr, 'weight_decay': 1e-2},
        {'params': rest_params, 'lr': args.lr, 'weight_decay': 1e-2}
    ])
    
    # 损失函数设置
    if args.focal_loss:
        criterion = FocalLoss(gamma=2)
        print("使用Focal Loss进行训练")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用标准交叉熵损失进行训练")
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # 训练过程跟踪
    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stop_counter = 0
    
    print('>> 开始训练')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, sample in enumerate(train_loader):
            images = sample['image'].to(device)
            questions = sample['question'].to(device)
            labels = sample['answer'].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            logits = model(images, questions)
            loss = criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 进度显示
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {train_loss/(batch_idx+1):.4f} "
                      f"Acc: {100.*train_correct/train_total:.2f}%")
        
        # 计算平均训练指标
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, sample in enumerate(val_loader):
                images = sample['image'].to(device)
                questions = sample['question'].to(device)
                labels = sample['answer'].to(device)
                
                # 前向传播
                logits = model(images, questions)
                loss = criterion(logits, labels)
                
                # 统计
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # 计算平均验证指标
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录日志
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")
        
        with open(os.path.join(save_dir, 'logs', 'train_log.txt'), 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_acc:.2f},{val_loss:.6f},{val_acc:.2f}\n")
        
        # 保存检查点
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(save_dir, 'checkpoints', f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"模型已保存到 {checkpoint_path}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(save_dir, 'best_acc.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"最佳准确率模型已保存: {best_val_acc:.2f}%")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, 'best_loss.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"最佳损失模型已保存: {best_val_loss:.4f}")
        
        # 早停
        if early_stop_counter >= args.patience:
            print(f"[!] 验证性能{args.patience}个epoch未提高，早停")
            break
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f">> 训练完成 | 训练时间:{training_time//60:.0f}分钟{training_time%60:.0f}秒")
    print(f">> 最佳验证准确率: {best_val_acc:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VQA模型小规模训练')
    
    # 数据相关参数
    parser.add_argument('--train_csv', type=str, default='data_proc/train.csv',
                       help='训练集CSV文件路径')
    parser.add_argument('--val_csv', type=str, default='data_proc/val.csv',
                       help='验证集CSV文件路径')
    parser.add_argument('--question_vocab', type=str, default='data_proc/question_vocabs.txt',
                       help='问题词汇表路径')
    parser.add_argument('--answer_vocab', type=str, default='data_proc/annotation_vocabs.txt',
                       help='答案词汇表路径')
    parser.add_argument('--sample_size', type=int, default=5000,
                       help='训练样本数量')
    
    # 模型相关参数
    parser.add_argument('--feature_size', type=int, default=1024,
                       help='特征大小')
    parser.add_argument('--word_embed', type=int, default=300,
                       help='词嵌入维度')
    parser.add_argument('--hidden_size', type=int, default=512,
                       help='LSTM隐藏层大小')
    parser.add_argument('--num_hidden', type=int, default=2,
                       help='LSTM层数')
    parser.add_argument('--max_qu_len', type=int, default=30,
                       help='最大问题长度')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小')
    parser.add_argument('--epochs', type=int, default=5,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='学习率')
    parser.add_argument('--cnn_lr', type=float, default=2e-5,
                       help='CNN主干学习率')
    parser.add_argument('--focal_loss', action='store_true',
                       help='是否使用Focal Loss')
    parser.add_argument('--save_every', type=int, default=1,
                       help='每隔多少个epoch保存一次')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载的工作线程数')
    parser.add_argument('--patience', type=int, default=3,
                       help='早停耐心值')
    
    # 输出目录
    parser.add_argument('--save_dir', type=str, default='runs/small_experiment',
                       help='模型保存目录')
    
    args = parser.parse_args()
    train_small_sample(args) 