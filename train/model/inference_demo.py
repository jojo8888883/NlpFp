import os
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from test_minimal import SimpleVQAModel

def inference_demo(args):
    """
    使用训练好的模型在单张图像上进行推理演示
    """
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleVQAModel(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 加载图像
    image = Image.open(args.image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 处理问题
    question_tokens = args.question.lower().split()[:20]  # 最大长度20 
    question_ids = np.zeros(20, dtype=np.int64)
    for i, token in enumerate(question_tokens):
        question_ids[i] = hash(token) % 10000  # 简单的哈希
    
    question_tensor = torch.tensor(question_ids).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(image_tensor, question_tensor)
        confidence, predicted = torch.softmax(outputs, dim=1).max(1)
        
    # 打印结果
    print(f"问题: {args.question}")
    print(f"预测类别: {predicted.item()}, 置信度: {confidence.item():.4f}")
    
    # 可视化
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"问题: {args.question}\n预测类别: {predicted.item()}, 置信度: {confidence.item():.4f}")
    plt.axis('off')
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"结果已保存至: {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VQA推理演示')
    
    parser.add_argument('--model_path', type=str, default='runs/minimal_test/model_minimal.pth',
                       help='模型路径')
    parser.add_argument('--image_path', type=str, default='001.jpg',
                       help='测试图像路径')
    parser.add_argument('--question', type=str, default='这个人在做什么?',
                       help='问题文本')
    parser.add_argument('--num_classes', type=int, default=100,
                       help='类别数量')
    parser.add_argument('--output_path', type=str, default='runs/minimal_test/result.png',
                       help='输出图像路径')
    
    args = parser.parse_args()
    inference_demo(args) 