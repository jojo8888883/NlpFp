import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ImgEncoder(nn.Module):
    """
    使用MobileNetV3作为图像编码器，减少内存消耗
    """
    def __init__(self, embed_dim):
        super(ImgEncoder, self).__init__()
        # 使用更小的模型来节省内存
        self.model = models.mobilenet_v3_small(pretrained=True)
        in_features = self.model.classifier[-1].in_features
        
        # 移除最后一层分类器
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        
        # 添加维度映射层
        self.fc = nn.Linear(in_features, embed_dim)
        
        # 冻结早期层
        for name, param in list(self.model.named_parameters())[:10]:
            param.requires_grad = False

    def forward(self, image):
        # 注意：不再使用with torch.no_grad()来允许反向传播
        img_feature = self.model(image)
        img_feature = self.fc(img_feature)
        # L2归一化但不再detach，允许梯度传播
        l2_norm = F.normalize(img_feature, p=2, dim=1)
        return l2_norm


class QuEncoder(nn.Module):
    """
    使用双向LSTM作为问题编码器
    """
    def __init__(self, qu_vocab_size, word_embed, hidden_size, num_hidden, qu_feature_size):
        super(QuEncoder, self).__init__()
        self.word_embedding = nn.Embedding(qu_vocab_size, word_embed)
        self.tanh = nn.Tanh()
        # 将LSTM改为双向
        self.lstm = nn.LSTM(
            word_embed, 
            hidden_size, 
            num_layers=num_hidden, 
            bidirectional=True, 
            batch_first=False
        )
        # 注意：双向LSTM会使隐藏状态维度翻倍
        self.fc = nn.Linear(2 * 2 * num_hidden * hidden_size, qu_feature_size)

    def forward(self, question):
        qu_embedding = self.word_embedding(question)                # (batchsize, qu_length=30, word_embed=300)
        qu_embedding = self.tanh(qu_embedding)
        qu_embedding = qu_embedding.transpose(0, 1)                 # (qu_length=30, batchsize, word_embed=300)
        _, (hidden, cell) = self.lstm(qu_embedding)                 # hidden: (2*num_layers, batchsize, hidden_size)
        
        # 连接所有隐藏状态
        qu_feature = torch.cat((hidden, cell), dim=2)               # (2*num_layers, batchsize, 2*hidden_size)
        qu_feature = qu_feature.transpose(0, 1)                     # (batchsize, 2*num_layers, 2*hidden_size)
        qu_feature = qu_feature.reshape(qu_feature.size()[0], -1)   # (batchsize, 2*2*num_layers*hidden_size)
        qu_feature = self.tanh(qu_feature)
        qu_feature = self.fc(qu_feature)                            # (batchsize, qu_feature_size=1024)

        return qu_feature


class VQAModel(nn.Module):
    """
    VQA模型主架构，使用特征级联而非元素乘法
    """
    def __init__(self, feature_size, qu_vocab_size, ans_vocab_size, word_embed, hidden_size, num_hidden):
        super(VQAModel, self).__init__()
        self.img_encoder = ImgEncoder(feature_size)
        self.qu_encoder = QuEncoder(qu_vocab_size, word_embed, hidden_size, num_hidden, feature_size)
        
        # 使用特征级联的方式融合，需要处理2倍的输入维度
        self.fusion = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, ans_vocab_size),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Linear(ans_vocab_size, ans_vocab_size)
        )

    def forward(self, image, question):
        # 获取图像特征和问题特征
        img_feature = self.img_encoder(image)               # (batchsize, feature_size=1024)
        qst_feature = self.qu_encoder(question)             # (batchsize, feature_size=1024)
        
        # 特征级联替代元素乘法
        combined_feature = torch.cat([img_feature, qst_feature], dim=1)  # (batchsize, feature_size*2)
        
        # 融合层
        fused_feature = self.fusion(combined_feature)       # (batchsize, feature_size)
        
        # 分类层
        logits = self.classifier(fused_feature)             # (batchsize, ans_vocab_size)

        return logits
    
    
class FocalLoss(nn.Module):
    """
    实现Focal Loss用于处理答案类别不平衡问题
    """
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
