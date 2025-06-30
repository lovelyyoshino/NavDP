from PIL import Image
from flask import Flask, request, jsonify
from policy_agent import NavDP_Agent  # 导入NavDP导航策略代理
import numpy as np
import cv2
import imageio
import time
import datetime
import json
import os

from PIL import Image, ImageDraw, ImageFont
import argparse

# 命令行参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8888, help="服务器监听端口号")
parser.add_argument("--checkpoint", type=str, default="./checkpoints/navdp-weights.ckpt", 
                   help="导航策略模型权重路径")
args = parser.parse_known_args()[0]

# 创建Flask应用实例
app = Flask(__name__)

# 全局变量定义
navdp_navigator = None  # 导航代理实例，延迟初始化
navdp_fps_writer = None  # 用于记录轨迹可视化视频的写入器

def visualize(goal, image, image_step, depth, depth_step):
    """
    可视化函数：生成包含目标位置、RGB图像和深度图的可视化图像
    
    参数:
        goal: 目标位置坐标
        image: RGB图像
        image_step: RGB图像的时间戳或步数
        depth: 深度图
        depth_step: 深度图的时间戳或步数
        
    返回:
        return_image: 包含文本注释的可视化图像
        
    处理流程:
    1. 将深度图归一化并转换为RGB格式
    2. 在图像上创建黑色区域作为文字背景
    3. 合并RGB图像和深度图
    4. 添加目标位置和时间戳文本标注
    5. 保存结果并返回
    """
    # 深度图归一化和转换为RGB
    vis_depth = np.tile(depth,(1,1,3))  # 将单通道深度图复制为3通道
    vis_depth = (vis_depth - vis_depth.min()) / (vis_depth.max() - vis_depth.min() + 1e-6)  # 归一化到[0,1]
    vis_depth = (vis_depth * 255).astype(np.uint8)  # 转换为8位RGB格式
    
    # 复制图像并创建文本显示区域
    vis_image = image.copy()
    vis_image[20:40,20:270] = 0  # 创建黑色文本背景区域
    
    # 创建合并的可视化图像并添加文本标注
    concat_image = Image.fromarray(np.concatenate((vis_image,vis_depth),axis=1))  # 水平拼接RGB和深度图
    image_draw = ImageDraw.Draw(concat_image)
    font = ImageFont.truetype('DejaVuSansMono.ttf',16)
    
    # 添加目标位置文本
    text1 = "PointGoal: {}".format(np.round(goal,decimals=1).tolist())
    image_draw.text((20,20),text1, font=font, fill=(255,255,255))
    
    # 添加RGB和深度时间戳
    text2 = str(image_step)
    image_draw.text((20,40),text2, font=font, fill=(255,255,255))
    
    text3 = str(depth_step)
    image_draw.text((20,60),text3, font=font, fill=(255,255,255))

    # 转换回NumPy数组并保存
    return_image = np.asarray(concat_image).copy()
    cv2.imwrite("navdp-pred.png",return_image)  # 保存最新的预测结果
    return return_image

@app.route("/navdp_reset",methods=['POST'])
def navdp_reset():
    """
    重置导航系统的端点
    
    功能:
    1. 根据请求参数初始化或重置导航代理
    2. 创建或重置轨迹可视化视频写入器
    
    请求参数(JSON):
    - intrinsic: 相机内参矩阵
    - stop_threshold: 停止阈值，控制导航终止条件
    - batch_size: 批处理大小
    
    技术细节:
    - 首次调用时创建NavDP_Agent实例，之后仅重置状态
    - 每次调用都会创建新的视频写入器，记录名称包含时间戳
    """
    global navdp_navigator,navdp_fps_writer
    
    # 从请求中获取参数
    intrinsic = np.array(request.get_json().get('intrinsic'))  # 相机内参矩阵
    threshold = np.array(request.get_json().get('stop_threshold'))  # 停止阈值
    batchsize = np.array(request.get_json().get('batch_size'))  # 批处理大小
    
    # 初始化或重置导航代理
    if navdp_navigator is None:
        # 首次调用，创建导航代理实例
        navdp_navigator = NavDP_Agent(
            intrinsic,            # 相机内参矩阵，用于3D投影
            image_size=224,       # 输入图像尺寸
            memory_size=8,        # 记忆模块大小，存储历史观测
            predict_size=24,      # 轨迹预测步数
            temporal_depth=16,    # 时间深度，影响时序建模能力
            heads=8,              # Transformer多头注意力头数
            token_dim=384,        # 特征向量维度
            stop_threshold=threshold,  # 停止阈值
            navi_model=args.checkpoint, # 模型权重路径
            device='cuda:0')      # 运行设备
        navdp_navigator.reset(batchsize)  # 初始化批次大小
    else:
        # 非首次调用，仅重置状态
        navdp_navigator.reset(batchsize)

    # 初始化或重置视频写入器
    if navdp_fps_writer is None:
        # 创建新的视频写入器，文件名包含当前时间戳
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        navdp_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time), fps=7)
    else:
        # 关闭旧的视频写入器并创建新的
        navdp_fps_writer.close()
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        navdp_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time), fps=7)
    
    return jsonify()  # 返回空的JSON响应

@app.route("/navdp_reset_env",methods=['POST'])
def navdp_reset_env():
    """
    重置特定环境实例的端点
    
    功能:
    - 使用环境ID重置导航代理的环境状态
    
    请求参数(JSON):
    - env_id: 环境ID
    
    这个函数通常用于多环境训练或测试场景
    """
    global navdp_navigator
    navdp_navigator.reset_env(int(request.get_json().get('env_id')))  # 使用指定环境ID重置环境
    return jsonify()

@app.route("/navdp_step_xy",methods=['POST'])
def navdp_step_xy():
    """
    执行一步有目标导航的端点
    
    功能:
    1. 处理上传的RGB和深度图像
    2. 根据指定目标执行导航规划
    3. 记录轨迹可视化并返回导航结果
    
    请求参数:
    - image: RGB图像文件
    - depth: 深度图像文件
    - goal_data: 目标坐标JSON数据(goal_x, goal_y)
    
    返回:
    - trajectory: 实际执行轨迹
    - all_trajectory: 所有考虑的可能轨迹
    - all_values: 各轨迹的评估值
    
    技术细节:
    - 使用多个计时点(phase1-4)评估各处理阶段的性能
    - 深度图像按10000缩放因子进行归一化
    - 支持批处理模式的导航计算
    """
    global navdp_navigator,navdp_fps_writer
    start_time = time.time()  # 开始计时
    
    # 获取请求中的图像和目标数据
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y,np.ones_like(goal_x)),axis=1)  # 构建目标向量[x,y,1]
    batch_size = goal.shape[0]  # 批处理大小
    
    phase1_time = time.time()  # 数据读取完成的计时点
    
    # 处理RGB图像
    image = Image.open(image_file.stream)# 打开图像文件流
    image = image.convert('RGB')# 转换为RGB格式
    image = np.asarray(image)# 转换为NumPy数组
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 转换为BGR格式(OpenCV默认格式)
    image = image.reshape((batch_size, -1, image.shape[1], 3))  # 重塑为批处理格式
    
    # 处理深度图像
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')  # 转换为16位整数格式
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0  # 从整数值转换为实际深度(米)
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))  # 重塑为批处理格式
    
    phase2_time = time.time()  # 图像预处理完成的计时点
    
    # 执行有目标导航步骤
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_pointgoal(goal,image,depth)
    
    phase3_time = time.time()  # 导航计算完成的计时点
    
    # 记录轨迹可视化
    navdp_fps_writer.append_data(trajectory_mask)
    
    phase4_time = time.time()  # 可视化记录完成的计时点
    
    # 输出各阶段耗时统计
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(
        phase1_time - start_time,     # 数据读取耗时
        phase2_time - phase1_time,    # 图像预处理耗时
        phase3_time - phase2_time,    # 导航计算耗时
        phase4_time - phase3_time,    # 可视化记录耗时
        time.time() - start_time))    # 总耗时
    
    # 返回导航结果
    return jsonify({
        'trajectory': execute_trajectory.tolist(),       # 实际执行轨迹
        'all_trajectory': all_trajectory.tolist(),       # 所有可能轨迹
        'all_values': all_values.tolist()               # 轨迹评估值
    })

@app.route("/navdp_step_nogoal",methods=['POST'])
def navdp_step_nogoal():
    """
    执行一步无目标导航的端点
    
    功能:
    与navdp_step_xy类似，但执行的是无目标探索导航
    即使请求中提供了goal数据，也会被忽略，导航算法将自主决定探索方向
    
    处理流程与navdp_step_xy基本相同，但调用的是navigator的step_nogoal方法
    """
    global navdp_navigator,navdp_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y,np.ones_like(goal_x)),axis=1)
    batch_size = goal.shape[0]
    
    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_nogoal(image,depth)
    phase3_time = time.time()
    navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))
    return jsonify({'trajectory': execute_trajectory.tolist(),
                    'all_trajectory': all_trajectory.tolist(),
                    'all_values': all_values.tolist()})

# 启动Flask应用服务器
app.run(host='127.0.0.1', port=args.port)  # 在本地指定端口运行服务器