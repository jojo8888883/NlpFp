import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from torchvision import transforms
from model import VQAModel
from build_dataset import Vocab, tokenizer

class GradCAM:
    """
    使用Grad-CAM技术可视化模型关注的图像区域
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        def forward_hook(module, input, output):
            self.activations = output
            return None
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            return None
        
        # 获取目标层
        if '.' in target_layer:
            layer_name = target_layer.split('.')
            module = self.model
            for name in layer_name:
                if name.isdigit():
                    module = module[int(name)]
                else:
                    module = getattr(module, name)
        else:
            module = getattr(self.model, target_layer)
            
        # 注册钩子
        self.hooks.append(module.register_forward_hook(forward_hook))
        self.hooks.append(module.register_backward_hook(backward_hook))
        
    def __call__(self, x, question, class_idx=None):
        # 前向传播
        self.model.eval()
        self.model.zero_grad()
        
        # 预测
        logits = self.model(x, question)
        
        # 获取预测类别
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
            
        # 目标类别的分数
        score = logits[:, class_idx].squeeze()
        
        # 反向传播
        score.backward()
        
        # 计算权重
        gradients = self.gradients.detach().cpu().numpy()[0]  # B, C, H, W
        activations = self.activations.detach().cpu().numpy()[0]  # B, C, H, W
        
        # 全局平均池化梯度
        weights = np.mean(gradients, axis=(1, 2))  # C
        
        # 加权组合激活图
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # H, W
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU
        cam = np.maximum(cam, 0)
        
        # 归一化
        if cam.max() > 0:
            cam = cam / cam.max()
            
        # 调整尺寸为输入图像大小
        cam = np.uint8(255 * cam)
        cam = np.uint8(Image.fromarray(cam).resize((x.shape[3], x.shape[2]), Image.LANCZOS))
        
        return cam
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def visualize_attention(args):
    """
    可视化模型在图像上的注意力
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载词汇表
    print(f"加载词汇表")
    qu_vocab = Vocab(args.question_vocab)
    ans_vocab = Vocab(args.answer_vocab)
    
    # 构建模型
    print(f"构建模型")
    model = VQAModel(
        feature_size=args.feature_size,
        qu_vocab_size=qu_vocab.vocab_size,
        ans_vocab_size=ans_vocab.vocab_size,
        word_embed=args.word_embed,
        hidden_size=args.hidden_size,
        num_hidden=args.num_hidden
    ).to(device)
    
    # 加载模型权重
    print(f"加载模型权重: {args.ckpt}")
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 加载图像
    image = Image.open(args.image).convert('RGB')
    orig_image = np.array(image)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 处理问题
    tokens = tokenizer(args.question)
    question = np.array([qu_vocab.word2idx('<pad>')] * args.max_qu_len)
    question[:min(len(tokens), args.max_qu_len)] = [
        qu_vocab.word2idx(token) for token in tokens[:args.max_qu_len]
    ]
    question_tensor = torch.tensor([question], dtype=torch.long).to(device)
    
    # 初始化Grad-CAM
    target_layer = "img_encoder.model.features.30"  # VGG19的conv5_4层
    grad_cam = GradCAM(model, target_layer)
    
    # 计算Grad-CAM
    cam = grad_cam(image_tensor, question_tensor)
    
    # 可视化
    plt.figure(figsize=(12, 4))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(orig_image)
    plt.title('原始图像')
    plt.axis('off')
    
    # 热力图
    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title('注意力热力图')
    plt.axis('off')
    
    # 原始图像 + 热力图
    plt.subplot(1, 3, 3)
    plt.imshow(orig_image)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title('叠加热力图')
    plt.axis('off')
    
    # 添加问题和预测的答案
    with torch.no_grad():
        logits = model(image_tensor, question_tensor)
        predicted = logits.argmax(dim=1).item()
        predicted_answer = ans_vocab.idx2word(predicted)
    
    plt.suptitle(f'问题: {args.question}\n预测答案: {predicted_answer}', fontsize=14)
    
    # 保存可视化结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"可视化结果已保存至: {args.output}")
    
    # 移除钩子
    grad_cam.remove_hooks()
    
    return {
        'question': args.question,
        'answer': predicted_answer,
        'output_path': args.output
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VQA模型注意力可视化')
    
    # 输入参数
    parser.add_argument('--image', type=str, required=True,
                        help='要分析的图像路径')
    parser.add_argument('--question', type=str, required=True,
                        help='与图像相关的问题')
    parser.add_argument('--output', type=str, required=True,
                        help='输出可视化图像的路径')
    
    # 模型相关参数
    parser.add_argument('--ckpt', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--question_vocab', type=str, default='../data_proc/question_vocabs.txt',
                        help='问题词汇表路径')
    parser.add_argument('--answer_vocab', type=str, default='../data_proc/annotation_vocabs.txt',
                        help='答案词汇表路径')
    parser.add_argument('--max_qu_len', type=int, default=30,
                        help='最大问题长度')
    
    # 模型架构参数
    parser.add_argument('--feature_size', type=int, default=1024,
                        help='特征大小')
    parser.add_argument('--word_embed', type=int, default=300,
                        help='词嵌入维度')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='LSTM隐藏层大小')
    parser.add_argument('--num_hidden', type=int, default=2,
                        help='LSTM层数')
    
    args = parser.parse_args()
    visualize_attention(args) 