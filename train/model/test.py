import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from model import VQAModel
from build_dataset import VQADataset, Vocab, tokenizer

def test(args):
    """
    使用训练好的模型对验证集或测试集进行推理，并生成结果JSON文件
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载词汇表
    print(f"加载词汇表：{args.question_vocab}和{args.answer_vocab}")
    qu_vocab = Vocab(args.question_vocab)
    ans_vocab = Vocab(args.answer_vocab)
    qu_vocab_size = qu_vocab.vocab_size
    ans_vocab_size = ans_vocab.vocab_size
    print(f"问题词汇表大小: {qu_vocab_size}, 答案词汇表大小: {ans_vocab_size}")
    
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 加载验证集数据
    print(f"加载数据集：{args.val_csv}")
    val_data = pd.read_csv(args.val_csv)
    
    # 准备数据集和数据加载器
    dataset = VQADataset(
        csv_path=args.val_csv,
        question_vocab_path=args.question_vocab,
        answer_vocab_path=args.answer_vocab,
        max_qu_len=args.max_qu_len,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 构建模型
    print(f"构建模型...")
    model = VQAModel(
        feature_size=args.feature_size,
        qu_vocab_size=qu_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed=args.word_embed,
        hidden_size=args.hidden_size,
        num_hidden=args.num_hidden
    ).to(device)
    
    # 加载模型权重
    print(f"加载模型权重: {args.ckpt}")
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    
    # 进行推理
    results = []
    all_correct = 0
    all_samples = 0
    
    print("开始推理...")
    for idx, sample in enumerate(tqdm(dataloader, desc="推理中")):
        image = sample['image'].to(device)
        question = sample['question'].to(device)
        question_id = sample['question_id'].tolist()
        
        with torch.no_grad():
            logits = model(image, question)
            predict = torch.softmax(logits, dim=1)
            
        # 获取最高置信度的答案
        _, predicted = predict.max(1)
        predicted = predicted.tolist()
        
        # 将索引转换为答案文本
        predicted_answers = [ans_vocab.idx2word(idx) for idx in predicted]
        
        # 保存结果
        ans_qu_pair = [{'answer': ans, 'question_id': id} for ans, id in zip(predicted_answers, question_id)]
        results.extend(ans_qu_pair)
        
        # 如果有标签，计算准确度
        if 'answer' in sample:
            labels = sample['answer'].to(device)
            correct = predicted.eq(labels.cpu().numpy()).sum()
            all_correct += correct
            all_samples += len(predicted)
            
        # 显示进度
        if (idx + 1) % 50 == 0:
            print(f"处理中: {(idx+1)*args.batch_size}/{len(dataset)}")
    
    # 如果有标签，显示总体准确度
    if all_samples > 0:
        print(f"总体准确度: {100.0 * all_correct / all_samples:.2f}%")
    
    # 保存结果
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(results, f)
    
    print(f"结果已保存到 {args.out_json}")
    print(f"总共生成了 {len(results)} 个问答对")
    
    # 返回结果字典，以便进一步处理
    return {
        'results': results,
        'accuracy': 100.0 * all_correct / all_samples if all_samples > 0 else None
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VQA模型推理')
    
    # 数据相关参数
    parser.add_argument('--val_csv', type=str, required=True,
                        help='验证/测试CSV文件路径')
    parser.add_argument('--question_vocab', type=str, default='../data_proc/question_vocabs.txt',
                        help='问题词汇表路径')
    parser.add_argument('--answer_vocab', type=str, default='../data_proc/annotation_vocabs.txt',
                        help='答案词汇表路径')
    
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
    
    # 推理相关参数
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载的工作线程数')
    
    # 模型和输出路径
    parser.add_argument('--ckpt', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--out_json', type=str, required=True,
                        help='输出JSON文件路径')
    
    args = parser.parse_args()
    test(args)