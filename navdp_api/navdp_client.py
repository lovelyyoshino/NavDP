import requests
import json
import numpy as np
import argparse
import cv2
import io
import time
import imageio
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description="A script to run navdp")
parser.add_argument(
    "--port", type=int, default=8888)
parser.add_argument(
    "--rgb_pkl", type=str, default="./eval_images/trajectory_002/002_images_dataset.pkl")
parser.add_argument(
    "--depth_pkl", type=str, default="./eval_images/trajectory_002/002_depths_dataset.pkl")
parser.add_argument(
    "--output_path", type=str, default="./visualize.mp4")
args_cli = parser.parse_args()

def pointnav_reset(intrinsic=None,stop_threshold=-4.0,batch_size=1,env_id=None):
    print("http://localhost:%d/navdp_reset"%args_cli.port)
    if env_id is None:
        url = "http://localhost:%d/navdp_reset"%args_cli.port
        response = requests.post(url,json={'intrinsic':intrinsic.tolist(),'stop_threshold':stop_threshold,'batch_size':batch_size})
    else:
        url = "http://localhost:%d/navdp_reset_env"%args_cli.port
        response = requests.post(url,json={'env_id':env_id})
    return response

def pointnav_step(goals,rgb_images,depth_images):
    concat_images = np.concatenate([img for img in rgb_images],axis=0)
    concat_depths = np.concatenate([img for img in depth_images],axis=0)

    url = "http://localhost:%d/navdp_step_xy"%args_cli.port
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
        'goal_x': goals[:,0].tolist(),
        'goal_y': goals[:,1].tolist()
        }),
        'depth_time':time.time(),
        'rgb_time':time.time(),
    }
    response = requests.post(url, files=files, data=data)
    trajectory = json.loads(response.text)['trajectory']
    all_trajectory = json.loads(response.text)['all_trajectory']
    all_value = json.loads(response.text)['all_values']
    return np.array(trajectory),np.array(all_trajectory),np.array(all_value)

def nogoal_step(rgb_images,depth_images):
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
        'goal_x': np.zeros((rgb_images.shape[0],)).tolist(),
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
    if len(depth.shape) == 3:
        depth = depth[:,:,0]
    filter_z,filter_x = np.where(depth>0)
    depth_values = depth[filter_z,filter_x]
    pixel_z = (-depth.shape[0] + filter_z  + intrinsic[1][2]) * depth_values / intrinsic[1][1]
    pixel_x = (filter_x - intrinsic[0][2])*depth_values / intrinsic[0][0]
    pixel_y = depth_values
    color_values = rgb[filter_z,filter_x]
    point_values = np.stack([pixel_y,-pixel_x,pixel_z],axis=-1)
    return filter_z,filter_x,depth_values,point_values,color_values
    
def bev_visualize(intrinsic,rgb_image,depth_image,exec_trajectory,trajectory_all,trajectory_values):
    vis_depth_image = (np.clip((depth_image/2000.0),0,1)*255).astype(np.uint8)
    vis_depth_image = np.tile(vis_depth_image[:,:,None],(1,1,3))
    rgbd_vis_image = np.concatenate((rgb_image,vis_depth_image),axis=1)
    _,_,_,points,_ = get_pointcloud_from_depth(rgb_image,depth_image/1000.0,intrinsic)
    points = points[(points[:,2]>points[:,2].min() + 0.4) & (points[:,2]<points[:,2].max() - 1.0)]
    
    plt.figure(figsize=(10,6))
    ax1 = plt.subplot(1,2,1)
    ax1.scatter(-points[:,1],points[:,0],s=2)
    ax1.plot(-exec_trajectory[:,1],exec_trajectory[:,0],color='b',label='exec')
    ax1.set_xlim(-6.0, 6.0)
    ax1.set_ylim(-6.0, 6.0)
    
    norm_values = np.clip(trajectory_values+1.0,0,1)
    colormap = cm.get_cmap('jet')
    ax2 = plt.subplot(1,2,2)
    ax2.scatter(-points[:,1],points[:,0],s=2)
    for i in range(trajectory_all.shape[0]):
        color = np.array(colormap(norm_values[i])) * 255.0
        ax2.plot(-trajectory_all[i,:,1],trajectory_all[i,:,0],color=color[:3]/255.0,linewidth=1.0)
    ax2.set_xlim(-6.0, 6.0)
    ax2.set_ylim(-6.0, 6.0)
    
    fig = plt.gcf()
    fig.canvas.draw()
    fig_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_array = fig_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig_array = cv2.cvtColor(fig_array, cv2.COLOR_RGB2BGR)
    fig_array = cv2.resize(fig_array, (rgbd_vis_image.shape[1], rgbd_vis_image.shape[0]))
    
    vis_image = np.concatenate((rgbd_vis_image,fig_array),axis=0)
    return vis_image    
    
intrinsic = np.array([[360.0,0.0,350.0],[0.0,360.0,230.0],[0,0,1]])
with open(args_cli.depth_pkl, 'rb') as f:
    depth_data = pickle.load(f)
with open(args_cli.rgb_pkl, 'rb') as f:
    image_data = pickle.load(f)
rgb_length = len(image_data.keys())
depth_length = len(depth_data.keys())
assert rgb_length == depth_length
image_writer = imageio.get_writer(args_cli.output_path, fps=10)
for i in range(rgb_length):
    img_name = list(image_data.keys())[i] 
    img = image_data[img_name] 
    dep_name = list(depth_data.keys())[i]
    dep = depth_data[dep_name]
    if i == 0:
        pointnav_reset(intrinsic=intrinsic)
    rgb_images = np.array([img])
    depth_images = np.array([dep])
    goals = np.array([[10.0,0.0]])
    trajectory,all_trajectory,all_value = pointnav_step(goals,rgb_images,depth_images)
    #trajectory,all_trajectory,all_value = nogoal_step(rgb_images,depth_images)
    vis_image = bev_visualize(intrinsic,img,dep,trajectory[0],all_trajectory[0],all_value[0])
    image_writer.append_data(cv2.cvtColor(vis_image,cv2.COLOR_BGR2RGB))
image_writer.close()