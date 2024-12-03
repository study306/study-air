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

    d = np.sqrt(x**2 + y**2)

    if d > l1 + l2:
        raise ValueError("No solution: the given target is unreachable.")
    if d < abs(l1 - l2):
        raise ValueError("No solution: the given target is inside the dead zone.")

    alpha = np.arctan2(y, x)
    beta = np.arccos((l1**2 + d**2 - l2**2) / (2 * l1 * d))
    theta1 = alpha + beta

    gamma = np.arccos((l1**2 + l2**2 - d**2) / (2 * l1 * l2))
    theta2 = np.pi - gamma

    return theta1, theta2

x = float(input("Enter the x-coordinate of the target point: "))
y = float(input("Enter the y-coordinate of the target point: "))
theta_total_deg = float(input("Enter the total angle of the end-effector with respect to the base (in degrees): "))
l1 = float(input("Enter the length of the first link: "))
l2 = float(input("Enter the length of the second link: "))

theta_total = np.radians(theta_total_deg)

try:
    theta1, theta2 = inverse_kinematics(x, y, theta_total, l1, l2)

    theta1_deg = np.degrees(theta1)
    theta2_deg = np.degrees(theta2)

    print(f"Joint angle θ1: {theta1_deg:.2f} degrees")
    print(f"Joint angle θ2: {theta2_deg:.2f} degrees")

    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)

    print(f"Position of joint 2: ({x1:.2f}, {y1:.2f})")
    print(f"Position of end-effector: ({x2:.2f}, {y2:.2f})")

    plt.figure(figsize=(6, 6))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.plot([0, x1], [0, y1], 'r', linewidth=3, label='Link 1')
    plt.plot([x1, x2], [y1, y2], 'b', linewidth=3, label='Link 2')
    plt.scatter([0, x1, x2], [0, y1, y2], c=['black', 'blue', 'green'], zorder=5)
    plt.scatter(x, y, c='red', marker='x', label='Target Point', zorder=5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Inverse Kinematics of a 2-DOF Robot Arm')
    plt.xlim([-l1 - l2 - 1, l1 + l2 + 1])
    plt.ylim([-l1 - l2 - 1, l1 + l2 + 1])
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

g = 9.81

m1 = float(input("Enter mass of Link 1 (kg): "))
m2 = float(input("Enter mass of Link 2 (kg): "))
m_load = float(input("Enter mass of Load (kg): "))
L1 = float(input("Enter length of Link 1 (m): "))
L2 = float(input("Enter length of Link 2 (m): "))

thetal_range = np.linspace(0, 180, 180)
theta2_range = np.linspace(0, 180, 180)

taul_values = []
tau2_values = []
Fg1_values = []
Fg2_values = []
F_ext_values = []

def calculate_forces_and_torques(thetal, theta2):

    thetal_rad = np.radians(thetal)
    theta2_rad = np.radians(theta2)

    Fg1 = m1 * g
    Fg2 = m2 * g
    F_ext = m_load * g

    tau2 = Fg2 * (L2 / 2) * np.sin(theta2_rad) + F_ext * L2 * np.sin(theta2_rad)

    taul = (
        Fg1 * (L1 / 2) * np.sin(thetal_rad)
        + Fg2 * L1 * np.sin(thetal_rad)
        + Fg2 * (L2 / 2) * np.sin(thetal_rad + theta2_rad)
        + F_ext * (L1 + L2) * np.sin(thetal_rad + theta2_rad)
    )

    return Fg1, Fg2, F_ext, taul, tau2

for thetal in thetal_range:
    for theta2 in theta2_range:
        Fg1, Fg2, F_ext, taul, tau2 = calculate_forces_and_torques(thetal, theta2)
        Fg1_values.append(Fg1)
        Fg2_values.append(Fg2)
        F_ext_values.append(F_ext)
        taul_values.append(taul)
        tau2_values.append(tau2)

taul_values = np.array(taul_values)
tau2_values = np.array(tau2_values)
Fg1_values = np.array(Fg1_values)
Fg2_values = np.array(Fg2_values)
F_ext_values = np.array(F_ext_values)

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(thetal_range, taul_values[:len(thetal_range)], label=r'$\tau_1$ (Torque at Joint 1)')
plt.xlabel(r'$\theta_1$ (degrees)')
plt.ylabel("Torque (Nm)")
plt.title("Torque at Joint 1 vs. Joint Angle")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(thetal_range, Fg1_values[:len(thetal_range)], label=r'$F_{g1}$ (Force on Link 1)')
plt.xlabel(r'$\theta_1$ (degrees)')
plt.ylabel('Force (N)')
plt.title('Gravitational Force on Link 1 vs. Joint Angle')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(theta2_range, tau2_values[:len(theta2_range)], label=r'$\tau_2$ (Torque at Joint 2)')
plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel("Torque (Nm)")
plt.title("Torque at Joint 2 vs. Joint Angle")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(theta2_range, Fg2_values[:len(theta2_range)], label=r'$F_{g2}$ (Force on Link 2)')
plt.plot(theta2_range, F_ext_values[:len(theta2_range)], label=r'$F_{\text{ext}}$ (Force on Load)')
plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel('Force (N)')
plt.title('Gravitational Forces on Link 2 and Load vs. Joint Angle')
plt.legend()

plt.tight_layout()
plt.show()
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

