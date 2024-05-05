#!/usr/bin/env python3

# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

from GUI.State import State
from objects.Map import Map
from objects.Alerts import Alerts
from objects.Button import Button
from objects.Button_Text import Button_Text
from objects.Table import Table
from objects.Camera import Camera
from math import pi
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import json
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry 
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
import tf
import math
import os
from std_srvs.srv import SetBool, SetBoolRequest
from geometry_msgs.msg import PoseWithCovarianceStamped
import time
import rosservice

class DashBoard(State):
    """
    Initialize a new instance of the class with various attributes.

    Args:
        game: The game object.
        window: The window object.
        pipeRecv (multiprocessing.Pipe): The pipe for receiving data.
        pipeSend (multiprocessing.Pipe): The pipe for sending data.
        speed (int, optional): The initial speed (default is 0).
        position (tuple, optional): The initial position (default is (0, 0)).
        battery (int, optional): The initial battery level (default is 100).
        lane_error (int, optional): The initial lane error (default is 0).
        steer (int, optional): The initial steering value (default is 0).

    """

    angle_change = 2
    steer_change = error_change = battery_dx = 1
    clicked = False

    def __init__(
        self,
        game,
        window,
        pipeRecv,
        pipeSend,
        speed=0,
        position=(0, 0),
        battery=100,
        lane_error=0,
        steer=0,
    ):
        super().__init__(game, window)
        self.pipeRecv = pipeRecv
        self.pipeSend = pipeSend
        self.battery = battery
        self.lane_error = lane_error
        self.list = [1, 20, 33, 55, 66]
        self.cursor_image = self.game.image.load("setup/images/cursor.png")
        self.cursor_image = self.game.transform.scale(self.cursor_image, (25, 110))
        self.cursor_pivot = (12, 12)
        self.angle = 0
        self.sem = True
        self.names = {"load": 0, "save": 0, "reset": 0}
        self.speed_image = self.game.image.load("setup/images/Speed_Meter.png")
        self.speed_image = self.game.transform.scale(self.speed_image, (300, 400))
        self.battery_color = (0, 255, 0)
        self.seconds_fadeaway = 3

        self.little_car = self.game.image.load("setup/images/little_car.png")
        self.little_car = self.game.transform.scale(self.little_car, (85, 132))

        self.steer = steer
        self.wheel = self.game.image.load("setup/images/wheel.png")
        self.wheel = self.game.transform.scale(self.wheel, (60, 60))
        self.wheel_pivot = (self.wheel.get_width() / 2, self.wheel.get_height() / 2)

        self.arrow = self.game.image.load("setup/images/arrow.png")
        self.arrow = self.game.transform.scale(self.arrow, (60, 60))
        self.arrow_pivot = (self.arrow.get_width() / 2, self.arrow.get_height() / 2)

        self.font_big = self.game.font.SysFont(None, 70)
        self.font_small = self.game.font.SysFont(None, 30)
        self.font_little = self.game.font.SysFont(None, 25)
        self.buttonAutonomEnable = True
        self.buttonSpeedEnable = True
        self.button = Button(
            200, 550, self.startcommand, self.game, self.main_surface, "autonom"
        )
        self.button2 = Button(
            400, 550, self.startcommand, self.game, self.main_surface, "speed"
        )
        self.map = Map(40, 30, self.game, self.main_surface, car_x=230, car_y=1920)
        # self.alerts = Alerts(20, 240, self.game, self.main_surface, 250)
        # self.table = Table(
        #     self.pipeSend,
        #     self.pipeRecv,
        #     550,
        #     10,
        #     self.game,
        #     self.main_surface,
        #     width=600,
        #     height=300,
        # )
        # self.camera = Camera(850, 350, self.game, self.main_surface)
        self.camera = Camera(750, 10, self.game, self.main_surface, width=800, height=700)
        # self.buttonSave = Button_Text(970, 315, self.game, self.main_surface, "Save")
        # self.buttonLoad = Button_Text(1045, 315, self.game, self.main_surface, "Load")
        # self.buttonReset = Button_Text(1120, 315, self.game, self.main_surface, "Reset")
        # self.objects = [self.map, self.alerts, self.table, self.camera]
        self.objects = [self.map, self.camera]

        self.name = 'car1'
        self.odomState = np.zeros(2)
        self.gpsState = np.zeros(2)
        self.ekfState = np.zeros(2)
        self.gpsValuesList = []
        self.ekfValuesList = []
        self.odomValuesList = []
        self.waypoints = None
        self.detected_cars = None

        self.yaw1 = 0.0
        self.yaw2 = 0.0
        self.yaw1List = []
        self.yaw2List = []
        self.accelList_x = []
        self.accelList_y = []

        self.groundTwist = np.zeros(2)
        self.measuredTwist = np.zeros(2)
        self.groundSpeedXList = []
        self.groundSpeedYList = []
        self.measuredSpeedXList = []
        self.measuredSpeedYList = []

        self.car_idx = None
        self.car_pose = None
        self.car_inertial = None

        self.bridge = CvBridge()
        self.cv_image = np.zeros((640, 480))
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        self.command_sub = rospy.Subscriber("/car1/command", String, self.callback2)
        # rospy.spin()
        # Subscribe to topics
        # self.localization_sub = rospy.Subscriber("/automobile/localisation", localisation, self.gps_callback, queue_size=3)
        # self.localization_sub = rospy.Subscriber("/automobile/localisation", localisation, self.gps_callback, queue_size=3)
        # self.model_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.gps_callback, queue_size=3)
        self.ekf_sub = rospy.Subscriber("/odometry/filtered", Odometry, self.ekf_callback, queue_size=3)
        # self.gmapping_sub = rospy.Subscriber("/chassis_pose", PoseWithCovarianceStamped, self.gmapping_callback, queue_size=3)
        # self.hector_sub = rospy.Subscriber("/poseupdate", PoseWithCovarianceStamped, self.hector_callback, queue_size=3)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=3)
        self.odom_sub = rospy.Subscriber("/gps", PoseWithCovarianceStamped, self.odom_callback, queue_size=3)
        self.imu1_sub = rospy.Subscriber("/car1/imu", Imu, self.imu1_callback, queue_size=3)
        self.waypoint_sub = rospy.Subscriber("/waypoints", Float32MultiArray, self.waypoint_callback, queue_size=3)
        self.cars_sub = rospy.Subscriber("/car_locations", Float32MultiArray, self.cars_callback, queue_size=3)

        self.numObj = 0
        self.detected_objects = np.zeros(7)
        self.confidence_thresholds = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.65, 0.65, 0.65, 0.65, 0.7, 0.75]
        self.class_names = ["oneway", "highwayentrance", "stopsign", "roundabout", "park", "crosswalk", "noentry", "highwayexit", "priority", "lights", "block", "pedestrian", "car"]
        self.sign_sub = rospy.Subscriber("/sign", Float32MultiArray, self.sign_callback, queue_size=3)

        print(os.path.dirname(os.path.realpath(__file__))+'/map2024.png')
        self.map1 = cv2.imread(os.path.dirname(os.path.realpath(__file__))+'/map2024.png')
        #shape is (8107, 12223, 3)
        # self.map = cv2.resize(self.map, (700, int(self.map.shape[0]/self.map.shape[1]*700)))
        self.map1 = cv2.resize(self.map1, (700, int(1/1.38342246*700)))

    def sign_callback(self, sign):
        if sign.data:
            self.numObj = len(sign.data) // 7
            if self.numObj == 1:
                self.detected_objects = np.array(sign.data)
                # print(self.detected_objects)
                return
            self.detected_objects = np.array(sign.data)#.reshape(-1, 7).T
            # print(self.detected_objects)
        else:
            self.numObj = 0

    def startcommand(self, start):
        print("service call")
        # Wait for the service to become available
        rospy.wait_for_service("/start_bool", timeout=5)

        # Create a proxy for the service
        try:
            # Create a service proxy
            service_proxy = rospy.ServiceProxy("/start_bool", SetBool)

            # Create a service request object
            request = SetBoolRequest()

            # Set the value of the request (True or False)
            request.data = start  # Set to True or False as needed

            # Call the service
            response = service_proxy(request)

            # Process the response
            # Do something with response
            if response.success:
                rospy.loginfo("Service call succeeded!")
            else:
                rospy.logerr("Service call failed!")

        except rospy.ServiceException as e:
            print("Service call failed:", e)

    def callback(self, data):
        """
        :param data: sensor_msg array containing the image in the Gazsbo format
        :return: nothing but sets [cv_image] to the usefull image that can be use in opencv (numpy array)
        """
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")

        for i in range(self.numObj):
            text = ""
            try:
                id = int(self.detected_objects[7*i+6])
            except:
                return
            if self.detected_objects[7*i+5] < self.confidence_thresholds[id]:
                continue
            text = f"{self.class_names[id]} {self.detected_objects[7*i+5] * 100:.1f}%"
            label_size, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            x = int(self.detected_objects[7*i])
            y = int(self.detected_objects[7*i+1]) - label_size[1] - baseLine
            if y < 0:
                y = 0
            if x + label_size[0] > self.cv_image.shape[1]:
                x = self.cv_image.shape[1] - label_size[0]

            cv2.rectangle(self.cv_image, (x, y), (x + label_size[0], y + label_size[1] + baseLine), (255, 255, 255), -1)
            cv2.putText(self.cv_image, text, (x, y + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.rectangle(self.cv_image, (int(self.detected_objects[7*i]), int(self.detected_objects[7*i+1])), (int(self.detected_objects[7*i+2]), int(self.detected_objects[7*i+3])), (255, 255, 0), 2)
            
        self.camera.change_frame(self.cv_image)
        # cv2.imshow("Frame preview", self.cv_image)
        # key = cv2.waitKey(1)

    def callback2(self, data):
        """
        Parse the received data and set speed or steer accordingly.

        Args:
            data: The received data containing action type and value.
        """
        # Extract action type and value from the received string
        data_str = data.data
        try:
            data_dict = json.loads(data_str)
            action = int(data_dict.get("action", "0"))  # Convert action to int
            value = float(data_dict.get("speed", 0.0) if action == 1 else data_dict.get("steerAngle", 0.0))
            
            # Set speed or steer based on the action type
            if action == 1:
                # Set speed
                print("Setting speed to:", value)
                self.angle = self.rad_to_degrees(-5.5 * value)
                # Call a method to set speed
            elif action == 2:
                # Set steer angle
                print("Setting steer angle to:", value)
                self.steer = -1 * value
                # Call a method to set steer angle
            else:
                print("Invalid action:", action)
        except ValueError as e:
            print("Error parsing JSON:", e)

    def waypoint_callback(self, data):
        self.waypoints = data.data
    def cars_callback(self, data):
        self.detected_cars = data.data
    # def gps_callback(self, data):
    #     # self.gpsState[0] = data.posA
    #     # self.gpsState[1] = 15.0 - data.posB
    #     if self.car_idx is None:
    #         try:
    #             self.car_idx = data.name.index(self.name)
    #         except ValueError:
    #             return
    #     self.car_pose = data.pose[self.car_idx]
    #     self.car_inertial = data.twist[self.car_idx]
    #     self.gpsState[0] = self.car_pose.position.x
    #     # self.gpsState[1] = 15+self.car_pose.position.y
    #     self.gpsState[1] = self.car_pose.position.y
    #     self.groundTwist[0] = self.car_inertial.linear.x
    #     self.groundTwist[1] = self.car_inertial.linear.y
    def ekf_callback(self, data):
        self.ekfState[0] = data.pose.pose.position.x
        self.ekfState[1] = data.pose.pose.position.y
        self.yaw2 = tf.transformations.euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])[2]
        self.measuredTwist[0] = data.twist.twist.linear.x
        self.measuredTwist[1] = data.twist.twist.linear.y
    def gmapping_callback(self, data):
        # self.ekfState[0] = data.pose.pose.position.x
        # self.ekfState[1] = data.pose.pose.position.y
        self.odomState[0] = data.pose.pose.position.x
        self.odomState[1] = data.pose.pose.position.y
    # def hector_callback(self, data):
    #     self.ekfState[0] = data.pose.pose.position.x + 11.71
    #     self.ekfState[1] = data.pose.pose.position.y +  1.895
    def odom_callback(self, data):
        self.odomState[0] = data.pose.pose.position.x
        self.odomState[1] = data.pose.pose.position.y
    def imu1_callback(self, imu):
        self.yaw1 = -tf.transformations.euler_from_quaternion([imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w])[2]
        self.accelList_x.append(imu.linear_acceleration.x)
        self.accelList_y.append(imu.linear_acceleration.y)

    def display(self):
        # print("displaying")
        img_map = self.map1.copy()
        
        # ekf
        img_map = cv2.arrowedLine(img_map, (int(self.ekfState[0]/20.541*self.map1.shape[1]),int((13.656-self.ekfState[1])/13.656*self.map1.shape[0])),
                    ((int((self.ekfState[0]+0.75*math.cos(self.yaw1))/20.541*self.map1.shape[1]),int((13.656- (self.ekfState[1]+0.75*math.sin(self.yaw1)))/13.656*self.map1.shape[0]))), color=(255,0,255), thickness=3)
        cv2.circle(img_map, (int(self.ekfState[0]/20.541*self.map1.shape[1]),int((13.656-self.ekfState[1])/13.656*self.map1.shape[0])), radius=6, color=(0, 0, 255), thickness=-1)

        # odom 
        img_map = cv2.arrowedLine(img_map, (int(self.odomState[0]/20.541*self.map1.shape[1]),int((13.656-self.odomState[1])/13.656*self.map1.shape[0])),
                    ((int((self.odomState[0]+0.75*math.cos(self.yaw1))/20.541*self.map1.shape[1]),int((13.656- (self.odomState[1]+0.75*math.sin(self.yaw1)))/13.656*self.map1.shape[0]))), color=(255,0,255), thickness=3)
        cv2.circle(img_map, (int(self.odomState[0]/20.541*self.map1.shape[1]),int((13.656-self.odomState[1])/13.656*self.map1.shape[0])), radius=6, color=(0, 255, 0), thickness=-1)
        # cv2.addText(img_map, "Odom", (int(self.odomState[0]/20.541*self.map1.shape[1]),int((13.656-self.odomState[1])/13.656*self.map1.shape[0])), "Arial", 1, (0, 255, 0))

        # display the waypoints
        if self.waypoints is not None:
            for i in range(0, len(self.waypoints), 8):
                cv2.circle(img_map, (int(self.waypoints[i]/20.541*self.map1.shape[1]),int((13.656-self.waypoints[i+1])/13.656*self.map1.shape[0])), radius=1, color=(0, 255, 255), thickness=-1)

        # display the detected cars
        if self.detected_cars is not None:
            for i in range(2, len(self.detected_cars), 2):
                cv2.circle(img_map, (int(self.detected_cars[i]/20.541*self.map1.shape[1]),int((13.656-self.detected_cars[i+1])/13.656*self.map1.shape[0])), radius=5, color=(255, 255, 0), thickness=-1)
        
        # print([int(self.ekfState[0]/20.541*self.map1.shape[1]),int((13.656-self.ekfState[1])/13.656*self.map1.shape[0])])
        # print(self.map1.shape)
        #shape is (8107, 12223, 3)
        # Convert the OpenCV image to RGB format
        map1_rgb = cv2.cvtColor(img_map, cv2.COLOR_BGR2RGB)
        self.map.new_coordinates(int(self.odomState[0]/20.541*self.map1.shape[1]),int((13.656-self.odomState[1])/13.656*self.map1.shape[0]), map1_rgb)
        self.map.update()

        # windowName = 'track'
        # cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(windowName, 753, int(753/1.38342246))
        # cv2.imshow(windowName, img_map)
        # cv2.waitKey(10)
        return img_map

    def blitRotate(self, surf, image, pos, originPos, angle):
        """
        Rotate an image and blit it onto a surface.

        Args:
            surf: The target surface where the rotated image will be blitted.
            image: The image to be rotated and blitted.
            pos (tuple): The position (x, y) where the rotated image will be blitted.
            originPos (tuple): The pivot point (x, y) around which the image will be rotated.
            angle (float): The angle in degrees by which the image will be rotated.
        """
        image_rect = image.get_rect(
            topleft=(pos[0] - originPos[0], pos[1] - originPos[1])
        )
        offset_center_to_pivot = self.game.math.Vector2(pos) - image_rect.center
        rotated_offset = offset_center_to_pivot.rotate(-angle)
        rotated_image_center = (pos[0] - rotated_offset.x, pos[1] - rotated_offset.y)
        rotated_image = self.game.transform.rotate(image, angle)
        rotated_image_rect = rotated_image.get_rect(center=rotated_image_center)
        surf.blit(rotated_image, rotated_image_rect)

    def continous_update(self):
        """
        Continuously update the class attributes based on received messages.

        This method listens for incoming messages on the `pipeRecv` pipe and updates
        the class attributes accordingly, depending on the message type.
        """
        #shape is (8107, 12223, 3)
        # self.map.new_coordinates(11223, 2500)
        # self.map.update()
        self.display()
        service_list = rosservice.get_service_list()
        if '/start_bool' in service_list:
            self.button.update(True)
            self.buttonAutonomEnable = True
            self.button2.update(True)
            self.buttonSpeedEnable = True
            # print("The service '/start_bool' exists.")
        else:
            self.button.update(False)
            self.buttonAutonomEnable = False
            self.button2.update(False)
            self.buttonSpeedEnable = False
            # print("The service '/start_bool' does not exist.")
        # self.alerts.setValues("cross")
        time.sleep(0.1)
        if self.pipeRecv.poll():
            msg = self.pipeRecv.recv()
            if msg["action"] == "steering":
                self.steer = -1 * msg["value"]
            # elif msg["action"] == "modImg":
            #     self.camera.change_frame(msg["value"])
            # elif msg["action"] == "map":
            #     terms_list = msg["value"].split()
            #     x = float(terms_list[1][0 : len(terms_list[1]) - 1]) * 150
            #     y = float(terms_list[3][0 : len(terms_list[3]) - 1]) * 150
            #     self.map.new_coordinates(x, y)
            #     self.map.update()
            # elif msg["action"] == "battery":
            #     self.battery = msg["value"]
            # elif msg["action"] == "engStart":
            #     self.table.addValueFromPI("Able to start", msg["value"])
            # elif msg["action"] == "engRunning":
            #     self.table.addValueFromPI("Engine running", msg["value"])
            # elif msg["action"] == "speed":
            #     self.angle = self.rad_to_degrees(-1 * msg["value"])
            elif msg["action"] == "roadOffset":
                self.little_car = self.game.transform.scale(
                    self.little_car, (85 + msg["value"], 132)
                )
            # elif msg["action"] == "emptyAll":
            #     self.camera.conn_lost()

    def updateTimers(self, timePassed):
        """
        Update timers associated with named actions.

        This method updates timers for named actions stored in the `names` dictionary.
        It subtracts the specified `timePassed` from the timers.

        Args:
            timePassed (float): The time passed, in seconds.
        """
        for key, value in self.names.items():
            self.names[key] = value - timePassed

    def set_text(self, text):
        """
        Set a timer for a named action.

        This method sets a timer for a named action specified by the `text` parameter.
        The timer is initially set to 3.0 seconds.

        Args:
            text (str): The name of the action.
        """
        self.names[text] = 3.0

    def update(self):
        """
        Update the class state.

        This method updates the class state by performing the following actions:
        1. Calls the superclass's update method using `super()`.
        2. Calls the `continous_update` method to process incoming messages and update attributes.
        3. Calls the `input` method to handle user input.
        4. Adjusts the `battery_color` attribute based on the current battery level.
        """
        super().update()
        self.continous_update()
        self.input()
        self.battery_color = (
            (100 - self.battery) * 255 / 100,
            (self.battery / 100) * 255,
            0,
        )

    def rad_to_degrees(self, angle):
        """
        Convert an angle from radians to degrees.

        Args:
            angle (float): The angle in radians to be converted.

        """
        converted = angle * 180 / pi
        return converted

    def deg_to_radians(self, angle):
        """
        Convert an angle from degrees to radians.

        Args:
            angle (float): The angle in degrees to be converted.

        """
        converted = angle * pi / 180
        return converted

    def draw(self):
        """
        Draw the graphical elements on the main surface.

        This method clears the main surface, draws operation success messages with fading,
        draws various objects, buttons, battery level, speed image, and the little car image.

        """
        self.main_surface.fill(0)
        for key, value in self.names.items():
            if value > 0:
                text_surface = self.font_small.render(
                    key + " operation successfully", True, (255, 255, 255)
                )
                text_surface.set_alpha(255 / self.seconds_fadeaway * value)
                self.main_surface.blit(text_surface, (550, 310))

        for object in self.objects:
            object.draw()
        if self.buttonAutonomEnable:
            self.button.draw()
        if self.buttonSpeedEnable:
            self.button2.draw()
        # self.buttonSave.draw()
        # self.buttonLoad.draw()
        # self.buttonReset.draw()
        # battery_show = self.font_small.render(
        #     str(self.battery) + "%", True, self.battery_color
        # )
        # self.main_surface.blit(battery_show, (480, 10))
        self.main_surface.blit(self.speed_image, (450, -25))

        self.game.draw.line(self.main_surface, (255, 255, 255), (530, 480), (530, 310))
        self.game.draw.line(self.main_surface, (255, 255, 255), (660, 480), (660, 310))
        self.main_surface.blit(self.little_car, (553 + self.lane_error, 350))
        # self.game.draw.arc(
        #     self.main_surface,
        #     self.battery_color,
        #     [260, 10, 280, 250],
        #     pi / 4 + (100 - self.battery) * (pi / 2) / 100,
        #     pi - pi / 4,
        #     25,
        # )

        self.blitRotate(
            self.main_surface,
            self.cursor_image,
            (600, 165),
            self.cursor_pivot,
            self.angle,
        )

        self.blitRotate(
            self.main_surface, self.arrow, (595, 320), self.arrow_pivot, 90 + self.steer
        )

        if -self.steer > 0:
            steer_show = self.font_little.render(
                "+" + str(-self.steer) + "°", True, (255, 255, 255)
            )
            self.main_surface.blit(steer_show, (625, 300))
        elif -self.steer <= 0:
            steer_show = self.font_little.render(
                str(-self.steer) + "°", True, (255, 255, 255)
            )
            self.main_surface.blit(steer_show, (537, 300))
        super().draw()
