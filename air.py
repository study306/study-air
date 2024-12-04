import streamlit as st

# Function to display code examples
def display_code(code: str):
    st.code(code, language='python')

# Code examples
rdd_operations_example_1 = '''
import numpy as np
import matplotlib.pyplot as plt

def forward_kinematics(thetal, theta2, l1, l2):
    x = l1 * np.cos(thetal) + l2 * np.cos(thetal + theta2)
    y = l1 * np.sin(thetal) + l2 * np.sin(thetal + theta2)
    return x, y

thetal_deg = float(input("Enter the angle of the first joint (in degrees): "))
theta2_deg = float(input("Enter the angle of the second joint (in degrees): "))
l1 = float(input("Enter the length of the first link: "))
l2 = float(input("Enter the length of the second link: "))

thetal = np.radians(thetal_deg)
theta2 = np.radians(theta2_deg)

x, y = forward_kinematics(thetal, theta2, l1, l2)

print(f"End-effector position: (x: {x:.2f}, y: {y:.2f})")

plt.figure(figsize=(6, 6))

x1 = l1 * np.cos(thetal)
y1 = l1 * np.sin(thetal)
x2 = x
y2 = y

plt.plot([0, x1], [0, y1], 'r', linewidth=3)
plt.plot([x1, x2], [y1, y2], 'b', linewidth=3)

plt.plot(0, 0, 'ko', markersize=10)
plt.plot(x1, y1, 'bo', markersize=10)
plt.plot(x2, y2, 'ro', markersize=10)

plt.xlim([-l1 - l2, l1 + l2])
plt.ylim([-l1 - l2, l1 + l2])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Two-Link Robot Arm")

plt.grid(True)

plt.show()
'''

rdd_operations_example_2 = '''
import numpy as np
import matplotlib.pyplot as plt

def inverse_kinematics(x, y, theta_total, l1, l2):
    """
    Calculates the joint angles for a 2-DOF robot arm to reach a given target point.

    Args:
        x: The x-coordinate of the target point.
        y: The y-coordinate of the target point.
        theta_total: The total angle of the end-effector with respect to the base.
        l1: Length of the first link.
        l2: Length of the second link.

    Returns:
        A tuple containing the joint angles (theta1, theta2).
    """
    # Calculate the distance from the origin to the target point
    d = np.sqrt(x**2 + y**2)

    # Check if the target is reachable
    if d > l1 + l2 or d < abs(l1 - l2):
        raise ValueError("Target is out of reach for the robot arm.")

    # Calculate the angle between the x-axis and the line connecting the origin to the target point
    alpha = np.arctan2(y, x)

    # Calculate the angle between the first link and the line connecting the origin to the target point
    beta = np.arccos((l1**2 + d**2 - l2**2) / (2 * l1 * d))

    # Calculate theta1 using the given total angle
    theta1 = alpha - beta

    # Calculate theta2 based on the total angle of the end-effector with respect to the base
    theta2 = theta_total - theta1

    return theta1, theta2

# Get input from the user
x = float(input("Enter the x-coordinate of the target point: "))
y = float(input("Enter the y-coordinate of the target point: "))
theta_total_deg = float(input("Enter the total angle of the end-effector with respect to the base (in degrees): "))

# Convert theta_total to radians
theta_total = np.radians(theta_total_deg)

# Link lengths
l1 = 1  # Length of the first link
l2 = 1  # Length of the second link

# Calculate the joint angles
try:
    theta1, theta2 = inverse_kinematics(x, y, theta_total, l1, l2)

    # Convert the joint angles to degrees for easier interpretation
    theta1_deg = np.degrees(theta1)
    theta2_deg = np.degrees(theta2)

    # Display the results
    print(f"Joint angle θ1: {theta1_deg:.2f} degrees")
    print(f"Joint angle θ2: {theta2_deg:.2f} degrees")

    # Calculate the position of each joint
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta_total)
    y2 = y1 + l2 * np.sin(theta_total)

    # Display the joint positions
    print(f"Position of joint 1: ({x1:.2f}, {y1:.2f})")
    print(f"Position of end-effector: ({x2:.2f}, {y2:.2f})")

    # Plot the robot arm
    plt.figure(figsize=(6, 6))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.plot([0, x1], [0, y1], 'r-', linewidth=3, label='Link 1')  # First link
    plt.plot([x1, x2], [y1, y2], 'g-', linewidth=3, label='Link 2')  # Second link
    plt.plot(x, y, 'bo', markersize=8, label='Target Point')  # Target point
    plt.scatter([0, x1, x2], [0, y1, y2], c='k', zorder=5)  # Joint positions
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Inverse Kinematics of a 2-DOF Robot Arm')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xticks(np.arange(-2, 2.5, 0.5))
    plt.yticks(np.arange(-2, 2.5, 0.5))
    plt.legend()
    plt.show()

except ValueError as e:
    print(e)
'''

sampling_filtering_example_1 = '''
import numpy as np
import matplotlib.pyplot as plt

def forward_kinematics_3dof(theta1, theta2, theta3, l1, l2, l3):

    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)

    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)

    x3 = x2 + l3 * np.cos(theta1 + theta2 + theta3)
    y3 = y2 + l3 * np.sin(theta1 + theta2 + theta3)

    return x3, y3

theta1 = float(input("Enter the angle of the first joint (in degrees): "))
theta2 = float(input("Enter the angle of the second joint (in degrees): "))
theta3 = float(input("Enter the angle of the third joint (in degrees): "))
l1 = float(input("Enter the length of the first link: "))
l2 = float(input("Enter the length of the second link: "))
l3 = float(input("Enter the length of the third link: "))

theta1 = np.radians(theta1)
theta2 = np.radians(theta2)
theta3 = np.radians(theta3)

x, y = forward_kinematics_3dof(theta1, theta2, theta3, l1, l2, l3)

print("End-effector position: ({:.2f}, {:.2f})".format(x, y))

plt.figure(figsize=(8, 6))
plt.title("3-DOF Robot Arm")
plt.xlabel("X")
plt.ylabel("Y")

x1 = l1 * np.cos(theta1)
y1 = l1 * np.sin(theta1)
x2 = x1 + l2 * np.cos(theta1 + theta2)
y2 = y1 + l2 * np.sin(theta1 + theta2)

plt.plot([0, x1], [0, y1], 'b-', linewidth=3, label="Link 1")
plt.plot([x1, x2], [y1, y2], 'g-', linewidth=3, label="Link 2")
plt.plot([x2, x], [y2, y], 'r-', linewidth=3, label="Link 3")

plt.plot(0, 0, 'ko', markersize=10)
plt.plot(x1, y1, 'ko', markersize=10)
plt.plot(x2, y2, 'go', markersize=10)
plt.plot(x, y, 'ro', markersize=10)

plt.xlim([-l1-l2-l3, l1+l2+l3])
plt.ylim([-l1-l2-l3, l1+l2+l3])

plt.grid(True)
plt.legend()


plt.show()
'''

sampling_filtering_example_2 = '''
import numpy as np
import matplotlib.pyplot as plt

def inverse_kinematics_3dof(x, y, theta_total, l1, l2, l3):

    x_prime = x - l3 * np.cos(theta_total)
    y_prime = y - l3 * np.sin(theta_total)

    d = np.sqrt(x_prime**2 + y_prime**2)
    if d > l1 + l2 or d < abs(l1 - l2):
        raise ValueError("Target is out of reach for the robot arm.")

    theta1 = np.arctan2(y_prime, x_prime) - np.arccos((l1**2 + d**2 - l2**2) / (2 * l1 * d))
    theta2 = np.arccos((l1**2 + l2**2 - d**2) / (2 * l1 * l2)) - np.pi
    theta3 = theta_total - (theta1 + theta2)

    return theta1, theta2, theta3

x = float(input("Enter the x-coordinate of the target point: "))
y = float(input("Enter the y-coordinate of the target point: "))
theta_total_deg = float(input("Enter the total angle of the end-effector with respect to the base (in degrees): "))

theta_total = np.radians(theta_total_deg)

l1 = float(input("Enter the length of the first link: "))
l2 = float(input("Enter the length of the second link: "))
l3 = float(input("Enter the length of the third link: "))

try:
    theta1, theta2, theta3 = inverse_kinematics_3dof(x, y, theta_total, l1, l2, l3)
    theta1_deg = np.degrees(theta1)
    theta2_deg = np.degrees(theta2)
    theta3_deg = np.degrees(theta3)

    print(f"Joint angle θ1: {theta1_deg:.2f} degrees")
    print(f"Joint angle θ2: {theta2_deg:.2f} degrees")
    print(f"Joint angle θ3: {theta3_deg:.2f} degrees")

    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    x3 = x2 + l3 * np.cos(theta1 + theta2 + theta3)
    y3 = y2 + l3 * np.sin(theta1 + theta2 + theta3)

    plt.figure(figsize=(8, 8))
    plt.plot([0, x1], [0, y1], 'r-', linewidth=3, label='Link 1')
    plt.plot([x1, x2], [y1, y2], 'g-', linewidth=3, label='Link 2')
    plt.plot([x2, x3], [y2, y3], 'b-', linewidth=3, label='Link 3')
    plt.scatter([0, x1, x2, x3], [0, y1, y2, y3], c='k', zorder=5)
    plt.plot(x, y, 'ro', markersize=8, label='Target Point')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Inverse Kinematics of a 3-DOF Robot Arm")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()
except ValueError as e:
    print(e)

print("Result: The inverse kinematics of the 3-DOF robot arm was computed successfully.")
'''

sampling_filtering_example_3 = '''
import numpy as np
import matplotlib.pyplot as plt

def inverse_kinematics_3dof(x, y, theta_total, l1, l2, l3):
    """
    Calculates the joint angles for a 3-DOF robot arm to reach a given target point.

    Args:
        x: The x-coordinate of the target point.
        y: The y-coordinate of the target point.
        theta_total: The total angle of the end-effector with respect to the base.
        l1: Length of the first link.
        l2: Length of the second link.
        l3: Length of the third link.

    Returns:
        A tuple containing the joint angles (theta1, theta2, theta3).
    """
    # Calculate the distance from the origin to the target point
    d = np.sqrt(x**2 + y**2)

    # Check if the target is reachable
    if d > l1 + l2 + l3 or d < abs(l1 - (l2 + l3)):
        raise ValueError(f"Target is out of reach. The arm can reach between {abs(l1 - (l2 + l3)):.2f} and {l1 + l2 + l3:.2f} units.")

    # Calculate the angle between the x-axis and the line connecting the origin to the target point
    alpha = np.arctan2(y, x)

    # Calculate the distance to the point that excludes the third link
    d_reduced = d - l3

    # Calculate the angle between the first link and the line connecting the origin to the reduced target point
    beta = np.arccos((l1**2 + d_reduced**2 - l2**2) / (2 * l1 * d_reduced))

    # Joint angles:
    # Theta1
    theta1 = alpha - beta

    # Theta2 (the second joint angle depends on the total angle and the calculated theta1)
    theta2 = np.arctan2(y - l1*np.sin(theta1), x - l1*np.cos(theta1)) - theta1

    # Theta3 (the total angle minus the angles for the first two joints)
    theta3 = theta_total - (theta1 + theta2)

    return theta1, theta2, theta3

# Set link lengths to cover a range up to 10 units
l1 = 3.33  # Length of the first link
l2 = 3.33  # Length of the second link
l3 = 3.33  # Length of the third link

# Maximum and minimum reach
max_reach = l1 + l2 + l3
min_reach = abs(l1 - (l2 + l3))

# Display input ranges to guide the user
print(f"The robot arm can reach between {min_reach:.2f} and {max_reach:.2f} units from the base.")
print("Please provide target coordinates within this range.")
print(f"Note: The end-effector angle should be between -180 and 180 degrees.")

# Get user input for the target coordinates and angle
while True:
    try:
        x = float(input(f"Enter the x-coordinate of the target point (range: {-max_reach} to {max_reach}): "))
        y = float(input(f"Enter the y-coordinate of the target point (range: {-max_reach} to {max_reach}): "))
        theta_total_deg = float(input("Enter the total angle of the end-effector with respect to the base (in degrees, range: -180 to 180): "))

        # Convert theta_total to radians
        theta_total = np.radians(theta_total_deg)

        # Check if the input is within valid range
        d = np.sqrt(x**2 + y**2)
        if d > max_reach or d < min_reach:
            raise ValueError(f"Target is out of reach. Please enter coordinates between {min_reach:.2f} and {max_reach:.2f} units.")
        if not (-180 <= theta_total_deg <= 180):
            raise ValueError("Angle out of range. Please enter an angle between -180 and 180 degrees.")

        # If input is valid, break the loop
        break
    except ValueError as e:
        print(f"Invalid input: {e}. Please try again.")

# Calculate the joint angles
try:
    theta1, theta2, theta3 = inverse_kinematics_3dof(x, y, theta_total, l1, l2, l3)

    # Convert the joint angles to degrees for easier interpretation
    theta1_deg = np.degrees(theta1)
    theta2_deg = np.degrees(theta2)
    theta3_deg = np.degrees(theta3)

    # Display the results
    print(f"Joint angle θ1: {theta1_deg:.2f} degrees")
    print(f"Joint angle θ2: {theta2_deg:.2f} degrees")
    print(f"Joint angle θ3: {theta3_deg:.2f} degrees")

    # Calculate the position of each joint
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    x3 = x2 + l3 * np.cos(theta1 + theta2 + theta3)
    y3 = y2 + l3 * np.sin(theta1 + theta2 + theta3)

    # Display the joint positions
    print(f"Position of joint 1: ({x1:.2f}, {y1:.2f})")
    print(f"Position of joint 2: ({x2:.2f}, {y2:.2f})")
    print(f"Position of end-effector: ({x3:.2f}, {y3:.2f})")

    # Plot the robot arm
    plt.figure(figsize=(6, 6))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.plot([0, x1], [0, y1], 'r-', linewidth=3, label='Link 1')  # First link
    plt.plot([x1, x2], [y1, y2], 'g-', linewidth=3, label='Link 2')  # Second link
    plt.plot([x2, x3], [y2, y3], 'b-', linewidth=3, label='Link 3')  # Third link
    plt.plot(x, y, 'bo', markersize=8, label='Target Point')  # Target point
    plt.scatter([0, x1, x2, x3], [0, y1, y2, y3], c='k', zorder=5)  # Joint positions
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Inverse Kinematics of a 3-DOF Robot Arm')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.xticks(np.arange(-10, 11, 1))
    plt.yticks(np.arange(-10, 11, 1))
    plt.legend()
    plt.show()

except ValueError as e:
    print(e)
'''

ros_set_up = '''
#!/bin/bash

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

sudo apt install curl -y
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

sudo apt update
sudo apt install ros-noetic-desktop-full -y
source /opt/ros/noetic/setup.bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential -y

sudo apt install python3-rosdep -y
sudo rosdep init
rosdep update
'''

pub_sub = '''
# Step 1: Create and Build a ROS Workspace
mkdir -p ~/ros_workspace/src
cd ~/ros_workspace
catkin_make
source devel/setup.bash

# Step 2: Create a ROS Package
cd ~/ros_workspace/src
catkin_create_pkg my_chatter rospy std_msgs

# Step 3: Write the Publisher Node
mkdir ~/ros_workspace/src/my_chatter/scripts
touch ~/ros_workspace/src/my_chatter/scripts/publisher_node.py

echo '#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

def publisher():
    rospy.init_node("publisher_node", anonymous=True)
    pub = rospy.Publisher("/chatter", String, queue_size=10)
    rate = rospy.Rate(1)  # 1 message per second

    while not rospy.is_shutdown():
        msg = "Hello from the publisher node!"
        rospy.loginfo(f"Publishing: {msg}")
        pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass' > ~/ros_workspace/src/my_chatter/scripts/publisher_node.py
        
chmod +x ~/ros_workspace/src/my_chatter/scripts/publisher_node.py

# Step 4: Write the Subscriber Node
touch ~/ros_workspace/src/my_chatter/scripts/subscriber_node.py

echo '#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

def callback(msg):
    rospy.loginfo(f"Received: {msg.data}")

def subscriber():
    rospy.init_node("subscriber_node", anonymous=True)
    rospy.Subscriber("/chatter", String, callback)
    rospy.spin()

if __name__ == "__main__":
    try:
        subscriber()
    except rospy.ROSInterruptException:
        pass' > ~/ros_workspace/src/my_chatter/scripts/subscriber_node.py
        
chmod +x ~/ros_workspace/src/my_chatter/scripts/subscriber_node.py

# Step 5: Build the Package
cd ~/ros_workspace
catkin_make
source devel/setup.bash

# Step 6: Run the Nodes
# Open a terminal and run roscore
roscore

# In a new terminal, run the publisher node
cd ~/ros_workspace
source devel/setup.bash
rosrun my_chatter publisher_node.py

# In another new terminal, run the subscriber node
cd ~/ros_workspace
source devel/setup.bash
rosrun my_chatter subscriber_node.py

# Step 7: Verify Output
# To list active topics
rostopic list

# To see messages on /chatter
rostopic echo /chatter
'''

# Streamlit App Layout
st.title("AIR Experiments")

# Sidebar for selecting experiments
selected_experiment = st.sidebar.selectbox(
    "Select an Experiment", 
    ["2 DOF Forward", "ros_setup", "2 Dof Inverse", 
     "3 Dof forward", "3 Dof Inverse", "Torque","publisher and Subscriber"]
)

# Display corresponding code
if selected_experiment == "2 DOF Forward":
    st.header("2 Dof Forward")
    display_code(rdd_operations_example_1)
elif selected_experiment == "2 Dof Inverse":
    st.header("2 Dof inverse")
    display_code(rdd_operations_example_2)
elif selected_experiment == "3 Dof forward":
    st.header("3 dof forward")
    display_code(sampling_filtering_example_1)
elif selected_experiment == "3 Dof Inverse":
    st.header("3 Dof Inverse")
    display_code(sampling_filtering_example_2)
elif selected_experiment == "Torque":
    st.header("Torque and Force")
    display_code(sampling_filtering_example_3)
elif selected_experiment == "ros_setup":
    st.header("ROS Setup code")
    display_code(ros_set_up)
elif selected_experiment == "publisher and Subscriber":
    st.header("Publisher and Subscriber")
    display_code(pub_sub)

