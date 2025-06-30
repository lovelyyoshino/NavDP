import torch
import numpy as np
import cv2
from PIL import Image
from matplotlib import colormaps as cm
from policy_network import NavDP_Policy_DPT

class NavDP_Agent:
    """
    NavDP导航代理类 - 基于深度学习的视觉导航代理
    
    核心功能:
    1. 使用Transformer架构处理RGB-D图像序列
    2. 基于视觉记忆和目标进行轨迹规划
    3. 生成多条候选轨迹并评估其价值
    4. 支持有目标导航和无目标探索两种模式
    """
    
    def __init__(self,
                 image_intrinsic,     # 相机内参矩阵，用于3D投影计算
                 image_size=224,      # 输入图像的标准化尺寸
                 memory_size=8,       # 视觉记忆队列长度，存储历史观测
                 predict_size=24,     # 预测轨迹的路径点数量
                 temporal_depth=16,   # 时序建模深度，影响历史信息利用
                 heads=8,             # Transformer多头注意力机制的头数
                 token_dim=384,       # 特征向量维度
                 stop_threshold=-2.0, # 停止阈值，轨迹价值低于此值时停止移动
                 navi_model = "./100.ckpt",  # 预训练模型权重路径
                 device='cuda:0'):    # 计算设备
        """
        初始化NavDP导航代理
        
        技术架构:
        - 基于Vision Transformer的多模态融合网络
        - 结合RGB图像、深度信息和目标位置的端到端学习
        - 使用循环记忆机制保持空间一致性
        """
        self.image_intrinsic = image_intrinsic  # 相机内参，用于2D-3D投影变换
        self.device = device
        self.predict_size = predict_size
        self.image_size = image_size
        self.stop_threshold = stop_threshold    # 决策阈值，控制导航行为
        self.memory_size = memory_size
        
        # 初始化核心策略网络
        self.navi_former = NavDP_Policy_DPT(image_size, memory_size, predict_size, 
                                           temporal_depth, heads, token_dim, device)
        # 加载预训练权重并设置为推理模式
        self.navi_former.load_state_dict(torch.load(navi_model, map_location=self.device))
        self.navi_former.to(self.device)
        self.navi_former.eval()
    
    def reset(self, batch_size):
        """
        重置代理状态，初始化批处理的记忆队列
        
        参数:
            batch_size: 批处理大小，支持同时处理多个环境实例
        """
        self.memory_queue = [[] for i in range(batch_size)]  # 为每个批次初始化空的记忆队列
        
    def reset_env(self, i):
        """
        重置特定环境实例的记忆状态
        
        参数:
            i: 环境实例索引
        """
        self.memory_queue[i] = []  # 清空指定环境的视觉记忆
    
    def project_trajectory(self, images, n_trajectories, n_values):
        """
        将3D轨迹投影到2D图像上进行可视化
        
        参数:
            images: 原始RGB图像，形状为[batch_size, height, width, 3]
            n_trajectories: 所有候选轨迹，形状为[batch_size, num_trajectories, steps, 3]
            n_values: 各轨迹的评估值，形状为[batch_size, num_trajectories]
            
        返回:
            trajectory_mask: 绘制了轨迹的可视化图像
            
        技术细节:
        1. 使用相机内参将3D轨迹点投影到2D图像坐标
        2. 根据轨迹价值使用颜色编码(jet colormap)
        3. 在图像上绘制连续的轨迹线段
        """
        trajectory_masks = []
        for i in range(images.shape[0]):
            trajectory_mask = np.array(images[i])  # 复制原始图像作为画布
            n_trajectory = n_trajectories[i,:,:,0:2]  # 提取XY坐标
            n_value = n_values[i]  # 获取轨迹评估值
            
            # 遍历每条轨迹进行可视化
            for waypoints, value in zip(n_trajectory, n_value):
                # 价值归一化和颜色映射
                norm_value = np.clip(-value*0.1, 0, 1)  # 将价值映射到[0,1]区间
                colormap = cm.get('jet')  # 使用jet颜色映射
                color = np.array(colormap(norm_value)[0:3]) * 255.0  # 转换为RGB颜色
                
                # 3D轨迹点预处理
                input_points = np.zeros((waypoints.shape[0], 3)) - 0.2  # 初始化3D点
                input_points[:,0:2] = waypoints  # 设置XY坐标
                input_points[:,1] = -input_points[:,1]  # Y轴翻转，适应相机坐标系
                
                # 使用相机内参进行3D到2D投影
                camera_z = images[0].shape[0] - 1 - self.image_intrinsic[1][1] * input_points[:,2] / (input_points[:,0] + 1e-8) - self.image_intrinsic[1][2]
                camera_x = self.image_intrinsic[0][0] * input_points[:,1] / (input_points[:,0] + 1e-8) + self.image_intrinsic[0][2]
                
                # 绘制轨迹线段
                for i in range(camera_x.shape[0]-1):
                    try:
                        # 检查投影点是否在图像范围内
                        if camera_x[i] > 0 and camera_z[i] > 0 and camera_x[i+1] > 0 and camera_z[i+1] > 0:
                            # 在图像上绘制线段
                            trajectory_mask = cv2.line(trajectory_mask,
                                                     (int(camera_x[i]),int(camera_z[i])),
                                                     (int(camera_x[i+1]),int(camera_z[i+1])),
                                                     color.astype(np.uint8).tolist(), 5)
                    except:
                        pass  # 忽略投影异常
            trajectory_masks.append(trajectory_mask)
        return np.concatenate(trajectory_masks, axis=1)  # 水平拼接所有批次的结果

    def process_image(self, images):
        """
        RGB图像预处理管道
        
        处理步骤:
        1. 等比例缩放保持长宽比
        2. 居中填充到标准尺寸
        3. 归一化到[0,1]范围
        
        技术要点:
        - 使用等比例缩放避免图像畸变
        - 黑色填充保持几何关系
        - 浮点数归一化适配神经网络输入
        """
        assert len(images.shape) == 4  # 确保输入格式为[batch, height, width, channels]
        H, W, C = images.shape[1], images.shape[2], images.shape[3]
        prop = self.image_size / max(H, W)  # 计算缩放比例，保持长宽比
        return_images = []
        
        for img in images:
            # 等比例缩放
            resize_image = cv2.resize(img, (-1,-1), fx=prop, fy=prop)
            
            # 计算填充尺寸，使图像居中
            pad_width = max((self.image_size - resize_image.shape[1])//2, 0)
            pad_height = max((self.image_size - resize_image.shape[0])//2, 0)
            
            # 使用零值填充到标准尺寸
            pad_image = np.pad(resize_image, ((pad_height,pad_height),(pad_width,pad_width),(0,0)), 
                              mode='constant', constant_values=0)
            
            # 最终调整到精确的目标尺寸
            resize_image = cv2.resize(pad_image, (self.image_size, self.image_size))
            resize_image = np.array(resize_image)
            resize_image = resize_image.astype(np.float32) / 255.0  # 归一化到[0,1]
            return_images.append(resize_image)
        return np.array(return_images)

    def process_depth(self, depths):
        """
        深度图像预处理管道
        
        处理步骤:
        1. 处理无效深度值(无穷大)
        2. 等比例缩放和填充
        3. 深度范围裁剪(0.1m-5.0m)
        
        技术要点:
        - 深度传感器的有效测量范围限制
        - 过近和过远的深度值对导航无意义
        - 保持与RGB图像相同的几何变换
        """
        assert len(depths.shape) == 4
        depths[depths==np.inf] = 0  # 将无穷大深度值设为0
        H, W, C = depths.shape[1], depths.shape[2], depths.shape[3]
        prop = self.image_size / max(H, W)
        return_depths = []
        
        for depth in depths:
            # 等比例缩放
            resize_depth = cv2.resize(depth, (-1,-1), fx=prop, fy=prop)
            
            # 居中填充
            pad_width = max((self.image_size - resize_depth.shape[1])//2, 0)
            pad_height = max((self.image_size - resize_depth.shape[0])//2, 0)
            pad_depth = np.pad(resize_depth, ((pad_height,pad_height),(pad_width,pad_width)), 
                              mode='constant', constant_values=0)
            
            # 调整到目标尺寸
            resize_depth = cv2.resize(pad_depth, (self.image_size, self.image_size))
            
            # 深度值裁剪，过滤无效范围
            resize_depth[resize_depth>5.0] = 0   # 超过5米的深度设为0
            resize_depth[resize_depth<0.1] = 0   # 小于10cm的深度设为0
            return_depths.append(resize_depth[:,:,np.newaxis])
        return np.array(return_depths)
    
    def process_pointgoal(self, goals):
        """
        目标点预处理 - 限制目标范围确保导航安全性
        
        技术约束:
        - X坐标(前进方向)限制在[0,10]米，防止后退
        - Y坐标(左右方向)限制在[-10,10]米，防止过度转向
        """
        clip_goals = goals.clip(-10, 10)  # 整体范围限制
        clip_goals[:,0] = np.clip(clip_goals[:,0], 0, 10)  # X方向只允许前进
        return clip_goals
    
    def step_pointgoal(self, goals, images, depths):
        """
        执行一步有目标导航
        
        核心流程:
        1. 图像和深度预处理
        2. 更新视觉记忆队列
        3. 调用神经网络进行轨迹规划
        4. 应用停止条件判断
        5. 生成可视化结果
        
        返回:
            - good_trajectory: 最优执行轨迹
            - all_trajectory: 所有候选轨迹
            - all_values: 轨迹评估值
            - trajectory_mask: 轨迹可视化图像
        """
        # 预处理输入数据
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        
        # 更新每个批次的视觉记忆队列
        input_images = []
        for i in range(len(self.memory_queue)):# 检测每一个batch size情况下观测图片的处理输入
            if len(self.memory_queue[i]) < self.memory_size:
                # 记忆队列未满，直接添加并用零填充
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                # 前向零填充到固定长度
                input_image = np.pad(input_image,((self.memory_size - input_image.shape[0],0),(0,0),(0,0),(0,0)))
            else:
                # 记忆队列已满，移除最旧的观测
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])    
                input_image = np.array(self.memory_queue[i])
            input_images.append(input_image)#输入所有的图片
        input_image = np.array(input_images)#合并所有的图片

        # 调试：保存当前输入图像序列
        cv2.imwrite("input_image.jpg", np.concatenate(self.memory_queue[0], axis=0)*255)
        
        # 准备网络输入
        input_depth = process_depths
        input_goals = self.process_pointgoal(goals)
        
        # 神经网络前向推理
        all_trajectory, all_values, good_trajectory, bad_trajectory = self.navi_former.predict_pointgoal_action(input_goals, input_image, input_depth)#作为输入，以及对应的图片队列，和当前的深度图，来完成轨迹预测
        
        # 停止条件判断：如果所有轨迹价值都很低，则停止移动
        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0:2] = good_trajectory[:,:,:,0:2] * 0.0  # 将轨迹置零
        
        # 生成轨迹可视化
        trajectory_mask = self.project_trajectory(images, all_trajectory, all_values) 
        return good_trajectory[:,0], all_trajectory, all_values, trajectory_mask
    
    def step_nogoal(self, images, depths):
        """
        执行一步无目标探索
        
        与step_pointgoal类似，但不使用目标信息
        适用于自主探索和建图场景
        """
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                input_image = np.pad(input_image,((self.memory_size - input_image.shape[0],0),(0,0),(0,0),(0,0)))
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])    
                input_image = np.array(self.memory_queue[i])
                
            input_images.append(input_image)
        input_image = np.array(input_images)
        
        cv2.imwrite("input_image.jpg",np.concatenate(self.memory_queue[0],axis=0)*255)
        input_depth = process_depths
        all_trajectory, all_values, good_trajectory, bad_trajectory = self.navi_former.predict_nogoal_action(input_image, input_depth)
        
        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0:2] = good_trajectory[:,:,:,0:2] * 0.0
        
        trajectory_mask = self.project_trajectory(images,all_trajectory,all_values) 
        return good_trajectory[:,0], all_trajectory, all_values, trajectory_mask

