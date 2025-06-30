import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from policy_backbone import *

class SinusoidalPosEmb(nn.Module):
    """
    正弦位置编码模块 - 用于扩散模型中的时间步编码
    
    功能:
    为扩散过程中的时间步t生成连续的位置编码，使模型能够区分不同的去噪阶段
    
    技术原理:
    使用正弦和余弦函数的组合，为每个时间步生成唯一的高维编码向量
    这种编码方式在Transformer和扩散模型中广泛使用
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 编码维度

    def forward(self, x):
        """
        将时间步转换为正弦位置编码
        
        参数:
            x: 时间步张量，形状为[batch_size]
        返回:
            位置编码，形状为[batch_size, dim]
        """
        device = x.device
        half_dim = self.dim // 2
        # 计算频率衰减因子
        emb = math.log(10000) / (half_dim - 1)# 这个操作公式对应的是:log(10000) / (half_dim - 1),通过log来完成缩放因子
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)# 这个操作公式对应的是频率的指数衰减，首先生成一个从 0 到 ( \text{half_dim} - 1 ) 的张量。然后这个张量与之前计算的 -emb 相乘，得到一个新的张量。最后，使用 torch.exp 计算这个张量的指数
        # 生成正弦和余弦编码，对张量完成扩展，对每一个时间步进行编码
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class NavDP_Policy_DPT(nn.Module):
    """
    NavDP策略网络 - 基于扩散概率模型的导航策略
    
    核心架构:
    1. RGBD编码器: 处理视觉输入和历史记忆
    2. 扩散模型: 生成多样化的轨迹候选
    3. 价值网络: 评估轨迹质量
    4. Transformer解码器: 序列建模和注意力机制
    
    技术创新点:
    - 结合扩散模型和强化学习进行轨迹规划
    - 使用多候选采样和价值排序选择最优路径
    - 支持有目标和无目标两种导航模式
    """
    
    def __init__(self,
                 image_size=224,      # 输入图像尺寸
                 memory_size=8,       # 视觉记忆长度
                 predict_size=24,     # 预测轨迹长度(路径点数)
                 temporal_depth=8,    # Transformer解码器层数
                 heads=8,             # 多头注意力头数
                 token_dim=384,       # 特征向量维度
                 channels=3,          # 输入图像通道数
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.memory_size = memory_size
        self.predict_size = predict_size
        self.temporal_depth = temporal_depth
        self.attention_heads = heads
        self.input_channels = channels
        self.token_dim = token_dim
        
        # 核心组件初始化
        self.rgbd_encoder = NavDP_RGBD_Backbone(image_size, token_dim, memory_size=memory_size, device=device)  # RGBD特征编码器
        self.point_encoder = nn.Linear(3, self.token_dim)  # 目标点编码器，将3D坐标映射到特征空间
        
        # Transformer解码器配置，输入是【batch, predict_size, token_dim】，输出也是【batch, predict_size, token_dim】
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model = token_dim,              # 特征维度
            nhead = heads,                    # 注意力头数
            dim_feedforward = 4 * token_dim,  # 前馈网络维度
            activation = 'gelu',              # 激活函数
            batch_first = True,               # 批次维度在前
            norm_first = True)                # 层归一化前置
        self.decoder = nn.TransformerDecoder(decoder_layer = self.decoder_layer,
                                             num_layers = self.temporal_depth)  # 多层解码器
        
        # 嵌入层
        self.input_embed = nn.Linear(3, token_dim)  # 动作序列嵌入(x,y,z坐标)，输入是【batch, predict_size, 3】，输出是【batch, predict_size, token_dim】
        
        # 位置编码
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, memory_size * 16 + 2)  # 条件位置编码，输入是【batch, memory_size * 16 + 2, token_dim】，输出是【batch, memory_size * 16 + 2, token_dim】
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)            # 输出位置编码

        # 时间和输出层
        self.time_emb = SinusoidalPosEmb(token_dim)  # 扩散时间步编码
        self.layernorm = nn.LayerNorm(token_dim)     # 层归一化，输入是【batch, predict_size, token_dim】，输出是【batch, predict_size, token_dim】
        self.action_head = nn.Linear(token_dim, 3)   # 动作预测头(输出x,y,z位移)
        self.critic_head = nn.Linear(token_dim, 1)   # 价值评估头
        
        # 扩散模型调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=10,           # 扩散步数(较少步数用于快速推理)
            beta_schedule='squaredcos_cap_v2', # 噪声调度策略
            clip_sample=True,                 # 样本裁剪
            prediction_type='epsilon')        # 预测噪声而非样本
        
        # 注意力掩码 - 确保自回归生成(当前步只能看到之前的步)
        self.tgt_mask = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = self.tgt_mask.float().masked_fill(self.tgt_mask == 0, float('-inf')).masked_fill(self.tgt_mask == 1, float(0.0))
        
        # 价值评估时的条件掩码 - 忽略目标信息，只基于视觉进行价值评估
        self.cond_critic_mask = torch.zeros((predict_size, 2 + memory_size * 16))
        self.cond_critic_mask[:,0:2] = float('-inf')  # 屏蔽前两个位置(时间和目标编码)
    
    def predict_noise(self, last_actions, timestep, goal_embed, rgbd_embed):
        """
        扩散模型的噪声预测网络
        
        功能:
        给定当前噪声轨迹、时间步和条件信息，预测需要去除的噪声
        这是扩散模型去噪过程的核心函数
        
        参数:
            last_actions: 当前噪声轨迹，形状[采样轨迹数量*目标batch, predict_size, 3]
            timestep: 扩散时间步
            goal_embed: 目标嵌入
            rgbd_embed: 视觉特征嵌入
            
        返回:
            预测的噪声，形状[batch, predict_size, 3]
        """
        # 动作序列嵌入和位置编码
        action_embeds = self.input_embed(last_actions)  # 将3D动作映射到特征空间
        time_embeds = self.time_emb(timestep.to(self.device)).unsqueeze(1).tile((last_actions.shape[0],1,1))  # 时间步编码。从[memory_size]映射到[1, token_dim]，然后扩展到[batch, 1, token_dim]

        # 条件信息组合：时间 + 目标 + 视觉特征，这个cond_pos_embed操作是为了能够让模型学习到这些信息在不同位置的影响
        cond_embedding = torch.cat([time_embeds, goal_embed, rgbd_embed], dim=1) + self.cond_pos_embed(torch.cat([time_embeds, goal_embed, rgbd_embed], dim=1))#所以对应的是[batch, memory_size * 16 + 2, token_dim]

        # 输入轨迹嵌入。将线性层和位置编码结合，然后把数据累加到一起作为当前
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        
        # Transformer解码器处理
        output = self.decoder(tgt = input_embedding,        # 目标序列
                             memory = cond_embedding,       # 记忆(条件信息)
                             tgt_mask = self.tgt_mask.to(self.device))  # 自回归掩码
        
        output = self.layernorm(output)
        output = self.action_head(output)  # 输出预测噪声
        return output
    
    def predict_critic(self, predict_trajectory, rgbd_embed):
        """
        价值网络 - 评估轨迹质量
        
        功能:
        基于预测轨迹和视觉特征，评估该轨迹的预期回报
        不使用目标信息，确保价值评估的通用性
        
        参数:
            predict_trajectory: 预测轨迹，形状[batch, predict_size, 3]
            rgbd_embed: 视觉特征嵌入
            
        返回:
            轨迹价值评分，形状[batch]
        """
        # 创建空目标嵌入(价值评估不依赖目标)
        nogoal_embed = torch.zeros_like(rgbd_embed[:,0:1])
        
        # 轨迹嵌入和位置编码
        action_embeddings = self.input_embed(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
        
        # 条件嵌入(不包含真实目标信息)
        cond_embeddings = torch.cat([nogoal_embed, nogoal_embed, rgbd_embed], dim=1) + \
                         self.cond_pos_embed(torch.cat([nogoal_embed, nogoal_embed, rgbd_embed], dim=1))
        
        # Transformer处理，使用掩码屏蔽目标信息
        critic_output = self.decoder(tgt = action_embeddings, 
                                    memory = cond_embeddings, 
                                    memory_mask = self.cond_critic_mask.to(self.device))
        
        critic_output = self.layernorm(critic_output)
        # 平均池化后输出标量价值，得到质量指标
        critic_output = self.critic_head(critic_output.mean(dim=1))[:,0]
        return critic_output
    
    def predict_pointgoal_action(self, goal_point, input_images, input_depths, sample_num=16):
        """
        有目标导航 - 生成朝向指定目标的轨迹
        
        核心流程:
        1. 编码视觉输入和目标信息
        2. 使用扩散模型生成多条候选轨迹
        3. 价值网络评估所有轨迹
        4. 选择最优和最差轨迹用于对比学习
        
        参数:
            goal_point: 目标点坐标，形状[batch, 3]
            input_images: RGB图像序列
            input_depths: 深度图像序列
            sample_num: 采样轨迹数量
            
        返回:
            all_trajectory: 所有生成轨迹
            critic_values: 轨迹价值评分
            positive_trajectory: 最优轨迹(top-2)
            negative_trajectory: 最差轨迹(bottom-2)
        """
        with torch.no_grad():  # 推理模式，不计算梯度
            # 输入编码
            tensor_point_goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self.device)# 大小为[batch, 3]的目标点坐标
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)  # 视觉特征编码，输入是【batch, 3, H, W】，输出是【batch, memory_size * 16, token_dim】
            pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)  # 目标点编码，将其从【batch, 3】通过linear层映射到【batch,token_dim】，然后unsqueeze(1)扩展到【batch, 1, token_dim】
    
            # 扩展到多个采样
            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)# 扩展视觉特征，将每个batch的视觉特征复制sample_num次
            pointgoal_embed = torch.repeat_interleave(pointgoal_embed, sample_num, dim=0)

            # 扩散模型生成过程，这里的goal_point.shape[0]对应的就是token_dim，sample_num对应的是16，然后predict_size对应的是24
            noisy_action = torch.randn((sample_num * goal_point.shape[0], self.predict_size, 3), device=self.device)  # 初始高斯噪声，数据形状为[batch * sample_num, predict_size, 3]
            naction = noisy_action
            
            # 设置扩散时间步，这里使用较少的步数以加快推理速度
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            
            # 逐步去噪生成轨迹
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction, k.unsqueeze(0), pointgoal_embed, rgbd_embed)  # 预测噪声，根据当前噪声轨迹、时间步、目标信息和视觉特征
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample  # 去噪一步
            
            # 价值评估
            critic_values = self.predict_critic(naction, rgbd_embed)
            critic_values = critic_values.reshape(goal_point.shape[0], sample_num)
            
            # 将动作序列转换为累积轨迹(位置序列)
            all_trajectory = torch.cumsum(naction / 4.0, dim=1)  # 缩放因子4.0控制步长
            all_trajectory = all_trajectory.reshape(goal_point.shape[0], sample_num, self.predict_size, 3)  # 将所有采样的轨迹转换为形状[batch, sample_num, predict_size, 3]

            # 选择最优轨迹(价值最高的前2条)
            sorted_indices = (-critic_values).argsort(dim=1)  # 降序排列
            topk_indices = sorted_indices[:,0:2]# 选择前2条最高价值轨迹
            batch_indices = torch.arange(goal_point.shape[0]).unsqueeze(1).expand(-1, 2)# 批次索引扩展，将每个batch的索引扩展到2个
            positive_trajectory = all_trajectory[batch_indices, topk_indices]
            
            # 选择最差轨迹(价值最低的前2条)
            sorted_indices = (critic_values).argsort(dim=1)   # 升序排列
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_point.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]
            
            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()
    
    def predict_nogoal_action(self, input_images, input_depths, sample_num=16):
        """
        无目标探索 - 生成自主探索轨迹
        
        与有目标导航类似，但使用零向量替代真实目标
        适用于环境探索和建图场景
        
        技术特点:
        - 不依赖外部目标信息
        - 基于内在好奇心和探索策略
        - 生成多样化的探索路径
        """
        with torch.no_grad():
            rgbd_embed = self.rgbd_encoder(input_images,input_depths)
            nogoal_embed = torch.zeros_like(rgbd_embed[:,0:1])
            rgbd_embed = torch.repeat_interleave(rgbd_embed,sample_num,dim=0)
            nogoal_embed = torch.repeat_interleave(nogoal_embed,sample_num,dim=0)
           
            noisy_action = torch.randn((sample_num * input_images.shape[0], self.predict_size, 3), device=self.device)
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction,k.unsqueeze(0),nogoal_embed,rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred,timestep=k,sample=naction).prev_sample
            
            critic_values = self.predict_critic(naction,rgbd_embed)
            critic_values = critic_values.reshape(input_images.shape[0],sample_num)
            
            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(input_images.shape[0],sample_num,self.predict_size,3)
            
            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(input_images.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]
            
            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(input_images.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]
            
            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()
