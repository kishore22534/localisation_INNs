#!/usr/bin/env python

import rospy

from geometry_msgs.msg import Pose2D, Twist
from gazebo_msgs.msg import ModelState


from torchvision import transforms

from gazebo_msgs.msg import ODEPhysics
from geometry_msgs.msg import Vector3
from tf.transformations import euler_from_quaternion
import csv
import numpy as np
import random

from geometry_msgs.msg import Pose 

import math

from gazebo_msgs.msg import ModelStates

from local_inn.srv import pose_communication_camera_6dof
from std_msgs.msg import Float32MultiArray

   

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

import csv

from mpl_toolkits.mplot3d import Axes3D 



import plotly.graph_objects as go
import plotly.io as pio

import statistics


plot_file = '/home/siva/cps_280_ws/src/local_inn/scripts/saved_3d_plot/interactive_3d_plot21thmay.html'
#csv_file = '/home/siva/gazebo_plugin_tutorial/6dof_test_path_data_6thmay.csv' 
csv_file = '/home/siva/gazebo_plugin_tutorial/test_trajectory_6dof_rpy_order_14th_may.csv'

pose_pub = rospy.Publisher('/estimated_pose', Pose2D, queue_size=1)



from scipy.spatial.transform import Rotation as R

def geodesic_distance(R1, R2):
    """
    Compute geodesic distance (in radians) between two rotation matrices.
    """
    R_rel = R1 @ R2.T
    trace = np.trace(R_rel)
    angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))  # Clip for numerical stability
    return angle

def compute_geodesic_errors(gt_eulers, pred_eulers, degrees=True, order='zyx'):
    """
    Compute geodesic errors between ground truth and predicted Euler angles.
    
    Args:
        gt_eulers: Nx3 array of ground truth Euler angles (yaw, pitch, roll)
        pred_eulers: Nx3 array of predicted Euler angles (yaw, pitch, roll)
        degrees: Whether input angles are in degrees
        order: Order of rotation axes (e.g., 'zyx' for yaw-pitch-roll)
        
    Returns:
        errors: Geodesic errors (in degrees)
    """
    gt_rot = R.from_euler(order, gt_eulers, degrees=degrees)
    pred_rot = R.from_euler(order, pred_eulers, degrees=degrees)

    errors = []
    for R_gt, R_pred in zip(gt_rot, pred_rot):
        angle_rad = geodesic_distance(R_gt.as_matrix(), R_pred.as_matrix())
        errors.append(np.degrees(angle_rad) if degrees else angle_rad)
    return np.array(errors)




fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(211, projection='3d')  # 3D subplot
ax2 = fig.add_subplot(212)  # 2D error plot

def update_plot_6dof(frame):
    if len(xy_error) == 0:
        return
    
    # --- First Plot: 3D Robot Positions ---
    ax1.clear()
    ax1.set_title("Robot Positions (3D)")
    ax1.set_xlim(-4.5, 4)
    ax1.set_ylim(-4.5, 4)
    ax1.set_zlim(-4.5, 4)  # Assuming z-range is similar

    ax1.plot3D(robot_pos_gt['x'], robot_pos_gt['y'], robot_pos_gt['z'], 'r-', label='GT_pose')
    ax1.plot3D(robot_pos_inferred['x'], robot_pos_inferred['y'], robot_pos_inferred['z'], 'b-', label='Inferred_pose')

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()

    # --- Second Plot: Errors ---
    ax2.clear()
    ax2.set_title("Position and orientation Errors")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Error")
    # ax2.set_xlim(0, len(xy_error))
    # ax2.set_ylim(0, 2)

    ax2.plot(xy_error, 'g-', label="Position Error")
    ax2.plot(theta_error, 'm-', label="Orientation Error")

    ax2.plot(yaw_vals, color='tab:blue', label="yaw vals")
    ax2.plot(pitch_vals, color='tab:orange', label="pitch vals")
    ax2.plot(roll_vals, color='tab:red', label="roll vals")

    

    max_xy_error = max(xy_error)
    avg_xy_error = sum(xy_error) / len(xy_error)
    median_xy_error = statistics.median(xy_error)

    max_theta_error = max(theta_error)
    avg_theta_error = sum(theta_error) / len(theta_error)

    ax2.text(0.95, 0.95, f"Max Pos Error: {max_xy_error:.2f}\nAvg Pos Error: {avg_xy_error:.2f}",  #avg_xy_error
             transform=ax2.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.5))
    ax2.text(0.95, 0.75, f"Max Geodesic Error: {max_theta_error:.2f}\nAvg Goedesic Error: {avg_theta_error:.2f}",
             transform=ax2.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.5))

    ax2.legend()


    # Create a 3D figure
    fig = go.Figure()

    # Ground truth trajectory
    fig.add_trace(go.Scatter3d(
        x=robot_pos_gt['x'],
        y=robot_pos_gt['y'],
        z=robot_pos_gt['z'],
        mode='lines',
        name='GT_pose',
        line=dict(color='red')
    ))

    # Inferred trajectory
    fig.add_trace(go.Scatter3d(
        x=robot_pos_inferred['x'],
        y=robot_pos_inferred['y'],
        z=robot_pos_inferred['z'],
        mode='lines',
        name='Inferred_pose',
        line=dict(color='blue')
    ))

    # Update layout
    fig.update_layout(
        title='Robot Positions (3D)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[-2, 3]),
            yaxis=dict(range=[-3, 1]),
            zaxis=dict(range=[3, -1])
        ),
        width=1200,
        height=1200,
    )

    pio.write_html(fig, file=plot_file, auto_open=False) 

    

def animate():
 
    ani = FuncAnimation(fig, update_plot_6dof, interval=100)
    plt.show()



########################################################


def image_publisher_client(prev_pose):
    rospy.wait_for_service('compute_inferred_pose_service_camera_6dof')

    try:
        # Create a service proxy to call the server
        compute_inferred_pose = rospy.ServiceProxy('compute_inferred_pose_service_camera_6dof', pose_communication_camera_6dof)

        # Call the se
        response = compute_inferred_pose(prev_pose)  # check if it is numpy array or python array

        # Print the inferred pose returned by the server
        return response.pose

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")


def get_gt_pose( ):
    model_states = rospy.wait_for_message('/gazebo/model_states', ModelStates)
    model_name ="volta"
    try:     
        
        # Find the index of the desired model
        if model_name in model_states.name:
            model_index = model_states.name.index(model_name)
            model_pose = model_states.pose[model_index]
            
            
            # Extract position, orientation, linear and angular velocity
            gt_position = model_pose.position
            orientation = model_pose.orientation

            quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)    
            # Convert quaternion to Euler angles
            roll, pitch, yaw = euler_from_quaternion(quaternion)

            if( yaw <0):
                yaw = yaw+ 2* math.pi                

            return gt_position.x, gt_position.y , yaw
            
        else:
            rospy.logerr(f"Model '{model_name}' not found in /gazebo/model_states")
            return None, None
    except rospy.ROSException as e:
        rospy.logerr(f"Error waiting for /gazebo/model_states: {e}")
        return None, None



robot_pos_gt = {'x': [], 'y': [], 'z':[]}
robot_pos_inferred= {'x': [], 'y': [], 'z':[]}

theta_error =[]
xy_error = []

yaw_vals=[]
pitch_vals =[]
roll_vals=[]

  

def camera_scan_loop():
    # initial_state = list(get_gt_pose())
    # initial_state[2] = initial_state[2] *180/math.pi
    # rospy.loginfo(f"intial state {initial_state}")
    #initial_state[2] = 2* math.pi -initial_state[2]
    #prev_pose = initial_state

    start = True

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            # Each row is a list of strings; convert to float if needed
            numbers = [float(value) for value in row]   
            #rospy.loginfo(f"nummbers are :{numbers}")

            if start == True:
                prev_pose = numbers[:6]  
                prev_pose[3] = numbers[3] * math.pi/180  #  roll, pitch, yaw in radians
                prev_pose[4] = numbers[4] * math.pi/180
                prev_pose[5] = numbers[5] * math.pi/180
                start =False


            gt_x, gt_y, gt_z = numbers[:3]            
            gt_yaw = numbers[3] * math.pi/180
            gt_pitch = numbers[4] * math.pi/180
            gt_roll = numbers[5] * math.pi/180

            #rospy.loginfo("6DOF_ sending request to NN model")
            x,y,z, yaw, pitch, roll = image_publisher_client(prev_pose)  
            #rospy.loginfo(F"6DOF_ Recieved response from NN model, {x},{y},{z}, {yaw}, {pitch}, {roll}")


            ################################# MODIFY THIS FOR 6DOF

            
            pose_msg = Pose2D()
            pose_msg.x = x
            pose_msg.y = y
            pose_msg.theta = z

            # Publish the message
            pose_pub.publish(pose_msg)
            
            ######################################

            robot_pos_inferred['x'].append(x)
            robot_pos_inferred['y'].append(y)   
            robot_pos_inferred['z'].append(z)

            
            #rospy.loginfo(f"YAW ground truth ={gt_theta} inferred value={theta_in_rad}")

            robot_pos_gt['x'].append( gt_x)
            robot_pos_gt['y'].append( gt_y)
            robot_pos_gt['z'].append(gt_z)

            #calculate error
            pos_err = math.sqrt((x-gt_x)**2+(y-gt_y)**2 + (z-gt_z)**2)
            xy_error.append(pos_err)
            if pos_err >0.4:
                print(f"error:{pos_err}, gt_values are {gt_x}, {gt_y}, {gt_z}")

            orient_error = compute_geodesic_errors(np.array([[yaw, pitch, roll]]), np.array([[gt_yaw,gt_pitch, gt_roll ]]), degrees=False)
            ##print(orient_error[0].item())

            theta_error.append(orient_error[0].item()) #orientation_error)
    
            prev_pose = [x,y,z, yaw, pitch, roll]

            yaw_vals.append(abs(yaw- gt_yaw))
            pitch_vals.append(abs(pitch- gt_pitch))
            roll_vals.append(abs(roll- gt_roll))




if __name__ == '__main__':

    rospy.init_node('camera_publisher_node', anonymous=True)       

    ros_thread = threading.Thread(target=camera_scan_loop)
    ros_thread.start()    

    animate()

