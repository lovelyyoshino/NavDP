import torch
import torch.nn as nn
from depth_anything.depth_anything_v2.dpt import DepthAnythingV2

class PositionalEncoding(nn.Module):
    """
    标准正弦位置编码 - Transformer的经典位置编码方案
    
    功能:
    为序列中的每个位置生成唯一的位置编码，使模型能够理解序列的顺序信息
    
    技术原理:
    使用不同频率的正弦和余弦函数组合，为每个位置生成固定的编码向量
    这种编码具有平移不变性和相对位置感知能力
    """
    def __init__(self, embed_dim, max_len=1000):
        super(PositionalEncoding, self).__init__()
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        
        # 计算频率衰减项，确保不同维度有不同的频率
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        # 偶数维度使用sin，奇数维度使用cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数索引
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数索引
        
        self.register_buffer('pe', pe)  # 注册为缓冲区，不参与梯度计算
        
    def forward(self, x):
        """返回对应序列长度的位置编码"""
        return self.pe[:x.size(1)]

class LearnablePositionalEncoding(nn.Module):
    """
    可学习位置编码 - 相比固定编码，能够适应特定任务
    
    优势:
    - 通过训练学习最优的位置表示
    - 更好地适应导航任务的序列特性
    - 能够编码更复杂的位置关系
    """
    def __init__(self, embed_dim, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        # 使用Embedding层学习位置编码，输入为[batch_size, seq_len]，输出为[batch_size, seq_len, embed_dim]，max_len是最大序列长度目前是8*16+2
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        
    def forward(self, x):
        """
        为输入序列生成可学习的位置编码
        
        参数:
            x: 输入序列，形状[batch_size, seq_len, embed_dim]，这里因为cat了，所以batch_size是1，seq_len是memory_size * 16 + 2，然后最后一个维度是embed_dim是3
        返回:
            位置编码，形状[batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        # 生成位置索引
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)  # [seq_len]
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]
        # 通过Embedding层获取位置编码
        position_encoding = self.position_embedding(position_ids)  # [batch_size, seq_len, embed_dim]
        return position_encoding

class NavDP_RGBD_Backbone(nn.Module):
    """
    NavDP的RGBD骨干网络 - 多模态视觉特征提取器
    
    核心功能:
    1. 使用预训练的DepthAnything模型处理RGB和深度图像
    2. 通过Transformer融合多帧时序信息
    3. 生成用于导航决策的高级语义特征
    
    技术架构:
    - 双分支处理：RGB分支 + 深度分支
    - 时序建模：处理历史帧序列
    - 特征融合：Transformer解码器整合多模态信息
    - 记忆压缩：生成固定长度的记忆表示
    """
    
    def __init__(self,
                 image_size=224,     # 输入图像尺寸
                 embed_size=512,     # 最终输出特征维度
                 memory_size=8,      # 记忆长度
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.memory_size = memory_size
        self.image_size = image_size
        self.embed_size = embed_size
        
        # DepthAnything模型配置 - 使用ViT-Small作为骨干网络
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        
        # RGB特征提取器 - 基于预训练的DepthAnything模型
        self.rgb_model = DepthAnythingV2(**model_configs['vits'])
        self.rgb_model = self.rgb_model.pretrained.float()  # 使用预训练权重
        self.rgb_model.eval()  # 冻结参数，只提取特征
        
        # ImageNet预处理标准化参数
        self.preprocess_mean = torch.tensor([0.485,0.456,0.406], dtype=torch.float32)  # RGB均值
        self.preprocess_std = torch.tensor([0.229,0.224,0.225], dtype=torch.float32)   # RGB标准差
        
        # 深度特征提取器 - 复用RGB模型架构处理深度信息
        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained.float()
        self.depth_model.eval()
        
        # Transformer组件用于时序特征融合
        self.former_query = LearnablePositionalEncoding(384, self.memory_size*16)  # 查询位置编码
        self.former_pe = LearnablePositionalEncoding(384, (self.memory_size+1)*256)  # 记忆位置编码,memory_size的rgb图像+1帧深度图像
        
        # 双层Transformer解码器 - 用于融合RGB和深度特征
        self.former_net = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(384, 8, batch_first=True),  # 单层配置：384维特征，8头注意力
            2)  # 2层解码器
        
        # 特征投影层 - 将384维特征映射到目标维度
        self.project_layer = nn.Linear(384, embed_size)
        
    def forward(self, images, depths):
        """
        前向传播 - 处理RGBD图像序列并生成导航特征
        
        处理流程:
        1. 分别处理RGB和深度图像
        2. 支持单帧(4D)和多帧时序(5D)输入
        3. 特征标准化和维度调整
        4. Transformer融合生成记忆表示
        
        参数:
            images: RGB图像，形状[B,H,W,C]或[B,T,H,W,C]
            depths: 深度图像，形状[B,H,W,1]或[B,T,H,W,1]
            
        返回:
            memory_token: 融合后的记忆特征，形状[B, memory_size*16, embed_size]
        """
        with torch.no_grad():  # 特征提取阶段不需要梯度
            
            # === RGB图像处理分支 ===
            if len(images.shape) == 4:  # 单帧模式
                # 调整维度顺序：BHWC -> BCHW (适配CNN输入格式)
                tensor_images = torch.as_tensor(images, dtype=torch.float32, device=self.device).permute(0,3,1,2)# 调整为[B, C, H, W]格式
                tensor_images = tensor_images.reshape(-1, 3, self.image_size, self.image_size)# 这个reshape是将单帧图像展平为[B, C, H, W]格式
                
                # ImageNet标准化，通过对RGB图像进行均值和标准差归一化，只是对每一个channel作处理
                tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1,3,1,1).to(self.device)) / \
                                   self.preprocess_std.reshape(1,3,1,1).to(self.device)
                # 通过预训练模型提取特征 - 获取中间层特征(更适合导航任务)
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]  # [C, H, W]

            elif len(images.shape) == 5:  # 多帧时序模式
                # 处理时序维度：BTHWC -> BTCHW，多了一个时间维度T
                tensor_images = torch.as_tensor(images, dtype=torch.float32, device=self.device).permute(0,1,4,2,3)
                B, T, C, H, W = tensor_images.shape
                
                # 将时序维度展平进行批量处理
                tensor_images = tensor_images.reshape(-1, 3, self.image_size, self.image_size)# [B*T, C, H, W]
                tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1,3,1,1).to(self.device)) / \
                                   self.preprocess_std.reshape(1,3,1,1).to(self.device)
                
                # 提取特征并重新组织为时序格式
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0].reshape(B, T*256, -1)# [B, T*256， 384]
            
            # === 深度图像处理分支 ===
            if len(depths.shape) == 4:  # 单帧深度
                # 深度图调整：BHWC -> BCHW，并复制到3通道(适配RGB模型)
                tensor_depths = torch.as_tensor(depths, dtype=torch.float32, device=self.device).permute(0,3,1,2)
                tensor_depths = tensor_depths.reshape(-1, 1, self.image_size, self.image_size)
                tensor_depths = torch.concat([tensor_depths, tensor_depths, tensor_depths], dim=1)  # 1通道->3通道
                
                # 使用深度模型提取特征
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
                
            elif len(depths.shape) == 5:  # 多帧深度序列
                tensor_depths = torch.as_tensor(depths, dtype=torch.float32, device=self.device).permute(0,1,4,2,3)
                B, T, C, H, W = tensor_depths.shape
                tensor_depths = tensor_depths.reshape(-1, 1, self.image_size, self.image_size)
                tensor_depths = torch.concat([tensor_depths, tensor_depths, tensor_depths], dim=1)
                
                # 时序深度特征提取
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0].reshape(B, T*256, -1)
            
            # === 多模态特征融合 ===
            # 连接RGB和深度特征，并添加位置编码
            former_token = torch.concat((image_token, depth_token), dim=1) + \
                          self.former_pe(torch.concat((image_token, depth_token), dim=1))
            
            # 创建查询向量 - 用于从融合特征中提取固定长度的记忆
            former_query = self.former_query(torch.zeros((image_token.shape[0], self.memory_size * 16, 384), device=self.device))
            
            # Transformer解码器融合 - 查询机制提取最重要的导航相关特征
            memory_token = self.former_net(former_query, former_token)
            
            # 投影到目标特征维度
            memory_token = self.project_layer(memory_token)
            
            return memory_token  # [B, memory_size*16, embed_size]