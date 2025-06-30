import requests
import json
import numpy as np
import argparse
import cv2
import io
import time
import imageio
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description="NavDP运行脚本")
parser.add_argument(
    "--port", type=int, default=8888, help="服务器端口号")
parser.add_argument(
    "--rgb_pkl", type=str, default="./eval_images/trajectory_002/002_images_dataset.pkl", 
    help="RGB图像数据的pickle文件路径")
parser.add_argument(
    "--depth_pkl", type=str, default="./eval_images/trajectory_002/002_depths_dataset.pkl",
    help="深度图像数据的pickle文件路径")
parser.add_argument(
    "--output_path", type=str, default="./visualize.mp4",
    help="输出可视化视频的路径")
args_cli = parser.parse_args()

def pointnav_reset(intrinsic=None,stop_threshold=-4.0,batch_size=1,env_id=None):
    """
    重置导航系统
    
    参数:
        intrinsic: 相机内参
        stop_threshold: 停止阈值
        batch_size: 批处理大小
        env_id: 环境ID
    
    返回:
        HTTP响应
    """
    print("http://localhost:%d/navdp_reset"%args_cli.port)
    if env_id is None:
        url = "http://localhost:%d/navdp_reset"%args_cli.port
        response = requests.post(url,json={'intrinsic':intrinsic.tolist(),'stop_threshold':stop_threshold,'batch_size':batch_size})
    else:
        url = "http://localhost:%d/navdp_reset_env"%args_cli.port
        response = requests.post(url,json={'env_id':env_id})
    return response

def pointnav_step(goals,rgb_images,depth_images):
    """
    执行导航步骤（含目标）- 向服务器发送图像和目标位置，获取导航轨迹
    
    参数:
        goals: 目标坐标数组，形状为[batch_size, 2]
        rgb_images: RGB图像数组，形状为[batch_size, height, width, 3]
        depth_images: 深度图像数组，形状为[batch_size, height, width]
    
    返回:
        trajectory: 实际执行的导航轨迹
        all_trajectory: 所有考虑过的可能轨迹
        all_value: 每条可能轨迹的评估值
    
    工作流程:
    1. 合并批次中的所有图像
    2. 将RGB图像编码为JPEG格式
    3. 将深度图像缩放并编码为PNG格式
    4. 通过HTTP POST请求发送到服务器
    5. 解析返回的JSON响应获取轨迹和评估值
    """
    # 合并图像批次
    concat_images = np.concatenate([img for img in rgb_images],axis=0)
    concat_depths = np.concatenate([img for img in depth_images],axis=0)

    url = "http://localhost:%d/navdp_step_xy"%args_cli.port
    
    # 将RGB图像编码为JPEG
    _, rgb_image = cv2.imencode('.jpg', concat_images)
    image_bytes = io.BytesIO()
    image_bytes.write(rgb_image)#设置
    
    # 将深度图像缩放并编码为PNG
    # 乘以10000并裁剪到16位整数范围内
    depth_image = np.clip(concat_depths*10000.0,0,65535.0).astype(np.uint16)
    _, depth_image = cv2.imencode('.png', depth_image)
    depth_bytes = io.BytesIO()
    depth_bytes.write(depth_image)
    
    # 准备HTTP请求的文件和数据
    files = {
        'image': ('image.jpg', image_bytes.getvalue(), 'image/jpeg'),
        'depth': ('depth.png', depth_bytes.getvalue(), 'image/png'),
    }
    data = {
        'goal_data': json.dumps({
        'goal_x': goals[:,0].tolist(),  # X坐标目标位置
        'goal_y': goals[:,1].tolist()   # Y坐标目标位置
        }),
        'depth_time':time.time(),  # 深度图时间戳
        'rgb_time':time.time(),    # RGB图像时间戳
    }
    
    # 发送HTTP POST请求并获取响应
    response = requests.post(url, files=files, data=data)
    
    # 解析JSON响应
    trajectory = json.loads(response.text)['trajectory']        # 实际执行轨迹
    all_trajectory = json.loads(response.text)['all_trajectory'] # 所有可能轨迹
    all_value = json.loads(response.text)['all_values']         # 轨迹评估值
    
    return np.array(trajectory),np.array(all_trajectory),np.array(all_value)

def nogoal_step(rgb_images,depth_images):
    """
    执行导航步骤（无目标）- 与pointnav_step类似，但使用零向量作为目标
    
    参数:
        rgb_images: RGB图像数组，形状为[batch_size, height, width, 3]
        depth_images: 深度图像数组，形状为[batch_size, height, width]
    
    返回:
        trajectory: 实际执行的导航轨迹
        all_trajectory: 所有考虑过的可能轨迹
        all_value: 每条可能轨迹的评估值
    
    说明:
    这个函数与pointnav_step相似，但使用零向量作为目标位置，适用于无目标探索场景
    """
    concat_images = np.concatenate([img for img in rgb_images],axis=0)
    concat_depths = np.concatenate([img for img in depth_images],axis=0)

    url = "http://localhost:%d/navdp_step_nogoal"%args_cli.port
    _, rgb_image = cv2.imencode('.jpg', concat_images)
    image_bytes = io.BytesIO()
    image_bytes.write(rgb_image)
    
    depth_image = np.clip(concat_depths*10000.0,0,65535.0).astype(np.uint16)
    _, depth_image = cv2.imencode('.png', depth_image)
    depth_bytes = io.BytesIO()
    depth_bytes.write(depth_image)
    
    files = {
        'image': ('image.jpg', image_bytes.getvalue(), 'image/jpeg'),
        'depth': ('depth.png', depth_bytes.getvalue(), 'image/png'),
    }
    data = {
        'goal_data': json.dumps({
        'goal_x': np.zeros((rgb_images.shape[0],)).tolist(),#这里就直接按照batch size来给出zero
        'goal_y': np.zeros((rgb_images.shape[0],)).tolist(),
        }),
        'depth_time':time.time(),
        'rgb_time':time.time(),
    }
    response = requests.post(url, files=files, data=data)
    trajectory = json.loads(response.text)['trajectory']
    all_trajectory = json.loads(response.text)['all_trajectory']
    all_value = json.loads(response.text)['all_values']
    return np.array(trajectory),np.array(all_trajectory),np.array(all_value)

def get_pointcloud_from_depth(rgb,depth,intrinsic):
    """
    从深度图像和RGB图像获取三维点云数据
    
    参数:
        rgb: RGB图像，形状为[height, width, 3]
        depth: 深度图像，形状为[height, width]或[height, width, 1]
        intrinsic: 相机内参矩阵，形状为[3, 3]
    
    返回:
        filter_z: 有效深度像素的Z坐标索引
        filter_x: 有效深度像素的X坐标索引
        depth_values: 有效像素的深度值
        point_values: 三维点云坐标，形状为[num_points, 3]
        color_values: 对应点云的RGB颜色值，形状为[num_points, 3]
    
    使用针孔相机模型将2D图像投影到3D空间
    """
    if len(depth.shape) == 3:
        depth = depth[:,:,0]
    filter_z,filter_x = np.where(depth>0)# 首先筛选出有效深度值(>0)的像素
    depth_values = depth[filter_z,filter_x]
    pixel_z = (-depth.shape[0] + filter_z  + intrinsic[1][2]) * depth_values / intrinsic[1][1]
    pixel_x = (filter_x - intrinsic[0][2])*depth_values / intrinsic[0][0]
    pixel_y = depth_values #根据相机内参矩阵将这些像素从图像坐标转换为3D世界坐标
    color_values = rgb[filter_z,filter_x] # 获取对应的RGB颜色值
    point_values = np.stack([pixel_y,-pixel_x,pixel_z],axis=-1)
    return filter_z,filter_x,depth_values,point_values,color_values
    
def bev_visualize(intrinsic,rgb_image,depth_image,exec_trajectory,trajectory_all,trajectory_values):
    """
    生成鸟瞰图(Bird's Eye View)可视化图像
    
    参数:
        intrinsic: 相机内参矩阵
        rgb_image: RGB图像
        depth_image: 深度图像
        exec_trajectory: 实际执行轨迹，形状为[steps, 2]
        trajectory_all: 所有考虑的轨迹，形状为[num_trajectories, steps, 2]
        trajectory_values: 各轨迹的评估值，形状为[num_trajectories]
    
    返回:
        vis_image: 可视化结果图像，包含RGB-D图像和鸟瞰图
    
    可视化内容:
    1. 上半部分: RGB图像和深度图像的并排显示
    2. 下半部分: 两个鸟瞰图面板
       - 左侧面板: 点云俯视图和实际执行轨迹
       - 右侧面板: 点云俯视图和所有考虑的轨迹(按评估值着色)
    """
    vis_depth_image = (np.clip((depth_image/2000.0),0,1)*255).astype(np.uint8) # 将深度图像归一化到0-255范围
    vis_depth_image = np.tile(vis_depth_image[:,:,None],(1,1,3)) # 将深度图像转换为RGB格式
    rgbd_vis_image = np.concatenate((rgb_image,vis_depth_image),axis=1) # 并排显示RGB和深度图像
    _,_,_,points,_ = get_pointcloud_from_depth(rgb_image,depth_image/1000.0,intrinsic) # 获取点云数据
    points = points[(points[:,2]>points[:,2].min() + 0.4) & (points[:,2]<points[:,2].max() - 1.0)] # 过滤掉过近和过远的点云
    
    # 绘制鸟瞰图
    plt.figure(figsize=(10,6))
    ax1 = plt.subplot(1,2,1)
    ax1.scatter(-points[:,1],points[:,0],s=2)
    ax1.plot(-exec_trajectory[:,1],exec_trajectory[:,0],color='b',label='exec')
    ax1.set_xlim(-6.0, 6.0)
    ax1.set_ylim(-6.0, 6.0)
    

    norm_values = np.clip(trajectory_values+1.0,0,1)# 将评估值归一化到0-1范围
    colormap = matplotlib.colormaps.get_cmap('jet')  # 使用新的方式获取颜色映射函数
    ax2 = plt.subplot(1,2,2)
    ax2.scatter(-points[:,1],points[:,0],s=2)# 绘制点云
    for i in range(trajectory_all.shape[0]):
        color = np.array(colormap(norm_values[i])) * 255.0
        ax2.plot(-trajectory_all[i,:,1],trajectory_all[i,:,0],color=color[:3]/255.0,linewidth=1.0)# 绘制轨迹
    ax2.set_xlim(-6.0, 6.0)
    ax2.set_ylim(-6.0, 6.0)
    
    fig = plt.gcf()
    fig.canvas.draw()
    fig_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)# 获取绘图数据
    fig_array = fig_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))# 将数据转换为图像数组(RGBA)
    fig_array = fig_array[:,:,:3]  # 只取RGB通道
    fig_array = cv2.cvtColor(fig_array, cv2.COLOR_RGB2BGR)
    fig_array = cv2.resize(fig_array, (rgbd_vis_image.shape[1], rgbd_vis_image.shape[0]))
    
    vis_image = np.concatenate((rgbd_vis_image,fig_array),axis=0)
    plt.close(fig)  # 关闭图形以释放内存
    return vis_image    
    
intrinsic = np.array([[360.0,0.0,350.0],[0.0,360.0,230.0],[0,0,1]])  # 相机内参矩阵
with open(args_cli.depth_pkl, 'rb') as f:
    depth_data = pickle.load(f)
with open(args_cli.rgb_pkl, 'rb') as f:
    image_data = pickle.load(f)
rgb_length = len(image_data.keys())
depth_length = len(depth_data.keys())
assert rgb_length == depth_length
image_writer = imageio.get_writer(args_cli.output_path, fps=20, format='mp4')  # 设置视频写入器，使用默认编码器
for i in range(rgb_length):
    img_name = list(image_data.keys())[i] 
    img = image_data[img_name] 
    dep_name = list(depth_data.keys())[i]
    dep = depth_data[dep_name]
    if i == 0:
        pointnav_reset(intrinsic=intrinsic)
    rgb_images = np.array([img])
    depth_images = np.array([dep])
    goals = np.array([[10.0,0.0]])  # 设置导航目标坐标为(10.0, 0.0)
    
    # 调用pointnav_step函数执行有目标导航，获取轨迹数据
    trajectory,all_trajectory,all_value = pointnav_step(goals,rgb_images,depth_images)
    
    # 下面这行被注释掉的代码是无目标导航的调用方式
    #trajectory,all_trajectory,all_value = nogoal_step(rgb_images,depth_images)
    
    # 使用bev_visualize函数生成鸟瞰图可视化结果
    # trajectory[0]表示第一批次的执行轨迹
    # all_trajectory[0]表示第一批次的所有考虑轨迹
    # all_value[0]表示第一批次的轨迹评估值
    vis_image = bev_visualize(intrinsic,img,dep,trajectory[0],all_trajectory[0],all_value[0])
    
    # 将BGR格式的可视化图像转换为RGB格式，并添加到视频写入器中
    image_writer.append_data(cv2.cvtColor(vis_image,cv2.COLOR_BGR2RGB))

# 处理完所有图像后，关闭视频写入器，完成视频生成
image_writer.close()