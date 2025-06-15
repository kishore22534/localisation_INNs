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

from local_inn.srv import pose_communication_camera
from std_msgs.msg import Float32MultiArray

   

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

import csv


pose_pub = rospy.Publisher('/estimated_pose', Pose2D, queue_size=1)

waypoint_list = [ (-0.38, -4.03), (-1.74, -1.15), (-1.25, 0.60), (2.06, 1.08), (4.38, -1.88), (6.08, -3.45), (7.45, -2.15), (4.98, -0.49), (1.98, -3.77)]

# Extracting waypoint coordinates
waypoints_x = [point[0] for point in waypoint_list]
waypoints_y = [point[1] for point in waypoint_list]



def update_plot_bk(frame):
    plt.clf()  # Clear the plot
    plt.plot(robot_pos_gt['x'], robot_pos_gt['y'], 'r-', label='GT_pose')  # Red line for Robot 1
    plt.plot(robot_pos_inferred['x'], robot_pos_inferred['y'], 'b-', label='Inferred_pose')  # Blue line for Robot 2
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Real-Time Robot Positions')
    plt.legend()
    plt.grid()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

def update_plot_bk(frame):

    # Update the first plot (robot positions)
    ax1.clear()
    ax1.set_title("Robot Positions")
    ax1.set_xlim(-8, 8)
    ax1.set_ylim(-8, 8)

    ax1.plot(robot_pos_gt['x'], robot_pos_gt['y'], 'r-', label='GT_pose')
    ax1.plot(robot_pos_inferred['x'], robot_pos_inferred['y'], 'b-', label='Inferred_pose')
    ax1.legend()

    # Update the second plot (errors)
    ax2.clear()
    ax2.set_title("Position and Yaw Errors")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Error")
    ax2.set_xlim(0, len(xy_error))  # Use indices as x-axis
    ax2.set_ylim(0, 1)  # Adjust based on error range
    ax2.plot(xy_error, 'g-', label="Position Error")
    ax2.plot(theta_error, 'm-', label="Yaw Error")
    ax2.legend()

def update_plot(frame):
    # Update the first plot (robot positions)
    if(len(xy_error)==0):
        return
    ax1.clear()
    ax1.set_title("Robot Positions")
    ax1.set_xlim(-8, 8)
    ax1.set_ylim(-8, 8)

    ax1.plot(robot_pos_gt['x'], robot_pos_gt['y'], 'r-', label='GT_pose')
    ax1.plot(robot_pos_inferred['x'], robot_pos_inferred['y'], 'b-', label='Inferred_pose')

    # Adding waypoints to the plot
    ax1.scatter(waypoints_x, waypoints_y, color='black', marker='x', label='Waypoints')
    ax1.legend()

    # Update the second plot (errors)
    ax2.clear()
    ax2.set_title("Position and yaw errors")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Error")
    ax2.set_xlim(0, len(xy_error))  # Use indices as x-axis
    ax2.set_ylim(0, 1)  # Adjust based on error range
    ax2.plot(xy_error, 'g-', label="Position Error")
    ax2.plot(theta_error, 'm-', label="Yaw Error")

    # Calculate max and average values
    max_xy_error = max(xy_error)
    avg_xy_error = sum(xy_error) / len(xy_error)
    max_theta_error = max(theta_error)
    avg_theta_error = sum(theta_error) / len(theta_error)

    # Display the max and average values on the plot
    ax2.text(0.95, 0.95, f"Max Pos Error: {max_xy_error:.2f}\nAvg Pos Error: {avg_xy_error:.2f}",
             transform=ax2.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.5))
    ax2.text(0.95, 0.75, f"Max yaw Error: {max_theta_error:.2f}\nAvg Yaw Error: {avg_theta_error:.2f}",
             transform=ax2.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.5))

    ax2.legend()



def animate_bk():
    fig = plt.figure()
    ani = FuncAnimation(fig, update_plot, interval=200)  # Update every 100ms
    plt.show()

def animate():
 
    ani = FuncAnimation(fig, update_plot, interval=100)
    plt.show()



########################################################


def image_publisher_client(prev_pose):
    rospy.wait_for_service('compute_inferred_pose_service_camera')

    try:
        # Create a service proxy to call the server
        compute_inferred_pose = rospy.ServiceProxy('compute_inferred_pose_service_camera', pose_communication_camera)

        # Call the service with the LIDAR data (send as list of floats)
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



robot_pos_gt = {'x': [], 'y': [], 'theta':[]}
robot_pos_inferred= {'x': [], 'y': [], 'theta':[]}

theta_error =[]
xy_error = []

csv_file = '/home/siva/scene_clutter_experiment/output_log.csv'

def camera_scan_loop():
    # initial_state = list(get_gt_pose())
    # initial_state[2] = initial_state[2] *180/math.pi
    # rospy.loginfo(f"intial state {initial_state}")
    #initial_state[2] = 2* math.pi -initial_state[2]
    #prev_pose = initial_state

    start = True

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        
        for row in reader:
            # Each row is a list of strings; convert to float if needed
            numbers = [float(value) for value in row]
            rospy.loginfo(f"nummbers are :{numbers}")

            if start == True:
                prev_pose = numbers[:3]
                prev_pose[2] = prev_pose[2] *180/math.pi
                start =False


            gt_x, gt_y, gt_theta = numbers[:3] #get_gt_pose()         #yaw is in radians   

            x,y,theta =image_publisher_client(prev_pose)    # here theta is in degrees ?

            #rospy.loginfo(f" data- {gt_x}, {gt_y}, {gt_theta}, {x}, {y},{theta}")

            theta_in_rad =   theta * math.pi/180

            #################################
            yaw_publish = theta_in_rad
            if( yaw_publish> math.pi):
                yaw_publish -=2*math.pi    
            
            pose_msg = Pose2D()
            pose_msg.x = x
            pose_msg.y = y
            pose_msg.theta = yaw_publish

            # Publish the message
            pose_pub.publish(pose_msg)
            
            ######################################

            robot_pos_inferred['x'].append(x)
            robot_pos_inferred['y'].append(y)        
            robot_pos_inferred['theta'].append(theta_in_rad)
            #rospy.loginfo(f"YAW ground truth ={gt_theta} inferred value={theta_in_rad}")

            robot_pos_gt['x'].append( gt_x)
            robot_pos_gt['y'].append( gt_y)
            robot_pos_gt['theta'].append(gt_theta)

            #calculate error
            pos_err = math.sqrt((x-gt_x)**2+(y-gt_y)**2)
            xy_error.append(pos_err)

            tmp =abs(theta_in_rad-gt_theta) 
            # if(tmp>2*math.pi):
            #     rospy.loginfo(f"unsual value of tmp={tmp}, gtruth={gt_theta}, inferred ={theta_in_rad}")
            orientation_error = min(tmp, (2* math.pi) - tmp)
            theta_error.append(orientation_error)
    
            #prev_pose = [x,y,theta]
            prev_pose = numbers[-3:]  # original scene previous pose



if __name__ == '__main__':

    rospy.init_node('camera_publisher_node', anonymous=True)       

    ros_thread = threading.Thread(target=camera_scan_loop)
    ros_thread.start()

    animate()

