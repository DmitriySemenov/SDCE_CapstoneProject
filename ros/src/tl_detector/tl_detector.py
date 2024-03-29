#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml
import numpy as np
import math
import time

STATE_COUNT_THRESHOLD = 2
USE_IMG_CLASSIF = True
SAVE_IMGS = False
SAVE_FREQ = 10
LOG_FREQ = 2 # Log frequency divider
MAX_WP_DIFF = 100
MIN_WP_DIFF = 1
MIN_WP_DIFF_YELLOW = 10
DEBUG_EN = True
LOOP_RATE = 5 # Loop rate in Hz
SCORE_THRESH = 0.4
DEBUG_SITE_BAG = False

TF_CLASS_GREEN = 1
TF_CLASS_RED = 2
TF_CLASS_YELLOW = 3
TF_CLASS_UNKNOWN = 4

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []
        self.has_image = False

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()

        tf_mdl_sim = rospy.get_param('~tf_mdl_sim')
        self.light_classifier = TLClassifier(tf_mdl_sim)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.proc_count = 0

        #rospy.spin()
        self.loop()

    def loop(self):
        """
            Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        """
        rate = rospy.Rate(LOOP_RATE)
        while not rospy.is_shutdown():

            light_wp, state, diff = self.process_traffic_lights()
            if (DEBUG_SITE_BAG):
                state = self.get_light_state(None, 75)
            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                # Stop on Red. 
                # Also, if light is yellow and car is far enough to stop, command stop
                # Also check if stop due to yellow is in progress and continue to stop 
                if (state == TrafficLight.RED or \
                    (state == TrafficLight.YELLOW and diff > MIN_WP_DIFF_YELLOW) or \
                        (state == TrafficLight.YELLOW and self.last_wp == light_wp)):
                    light_wp = light_wp
                else: 
                    light_wp = -1
                    
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1

            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoint_tree:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                                 waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """
        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_closest_wp_idx_ahead(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind the vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def get_light_state(self, light, diff):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light_class = self.last_state

        if (diff < MAX_WP_DIFF and diff > MIN_WP_DIFF):
            if(self.has_image):
                cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            if (USE_IMG_CLASSIF):
                if(not self.has_image):
                    self.prev_light_loc = None
                    return False
                    
                #Get classification
                red_count = 0
                green_count = 0
                yellow_count = 0
                light_count = 0
                [boxes, scores, classes, num] = self.light_classifier.get_classification(cv_image)
                
                for i in range(num):
                    if scores[i] >= SCORE_THRESH:
                        light_count += 1
                        if classes[i] == TF_CLASS_RED:
                            red_count += 1
                        elif classes[i] == TF_CLASS_GREEN:
                            green_count += 1
                        elif classes[i] == TF_CLASS_YELLOW:
                            yellow_count += 1

                if (light_count >= 1):
                    if red_count >= 1:
                        light_class = TrafficLight.RED
                    elif green_count >= 1:
                        light_class = TrafficLight.GREEN
                    elif yellow_count >= 1:
                        light_class = TrafficLight.YELLOW

                if (DEBUG_EN and (self.proc_count % LOG_FREQ == 0)):
                    rospy.logwarn("TF: Class={}, Light#={}, Red#={}, Yellow#={}, Green#={}".format(light_class, light_count, red_count, yellow_count, green_count))
            else:
                light_class = light.state

        #Save some image data for training, if close enough
        if (SAVE_IMGS and self.has_image and (self.proc_count % SAVE_FREQ == 0) and diff < MAX_WP_DIFF and diff > MIN_WP_DIFF):
            if (DEBUG_SITE_BAG):
                save_file = "../../../imgs/Real/{}-{:.0f}.jpeg".format('site', (time.time() * 100))
            else:
                save_file = "../../../imgs/Sim/{}-{:.0f}.jpeg".format(self.light_to_string(light_class), (time.time() * 100))
            
            cv2.imwrite(save_file, cv_image)

        return light_class

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        if(self.pose and self.waypoint_tree):
            car_wp_idx = self.get_closest_wp_idx_ahead()

            #Find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        
        if closest_light:
            self.proc_count += 1
            state = self.get_light_state(closest_light, diff)
            
            #if (DEBUG_EN and (self.proc_count % LOG_FREQ == 0)):
            #    rospy.logwarn("CarIdx={}, StopLineIdx={}, diff={}, state={}".format(car_wp_idx, line_wp_idx, diff, self.light_to_string(state)))
            
            return line_wp_idx, state, diff

        return -1, TrafficLight.UNKNOWN, 9999

    def light_to_string(self, light):
        light_str = "unknown"
        if light == TrafficLight.GREEN:
            light_str = "green"
        elif light == TrafficLight.YELLOW:
            light_str = "yellow"
        elif light == TrafficLight.RED:
            light_str = "red"
        return light_str

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
