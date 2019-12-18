#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 60 # Number of waypoints we will publish. You can change this number
LOOP_RATE = 10 # Loop rate in Hz
MAX_DECEL = 8
TGT_DECEL = 2
MIN_DIST_DECEL_LIM = 5

DEBUG_EN = False
LOG_FREQ = 10


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.traffic_waypoint_idx = -1
        self.proc_count = 0

        self.loop()

    def loop(self):
        rate = rospy.Rate(LOOP_RATE)
        while not rospy.is_shutdown():
            if self.pose and self.waypoint_tree:
                self.proc_count += 1
                #get closest waypoint
                closest_waypoint_idx = self.get_closest_wp_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def get_closest_wp_idx(self):
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
            #rospy.loginfo("closest_idx={}".format(closest_idx))
        return closest_idx
    
    def publish_waypoints(self, closest_idx):
        final_lane = self.generate_lane(closest_idx)
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self, closest_idx):
        lane = Lane()
        lane.header = self.base_waypoints.header

        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_wps = self.base_waypoints.waypoints[closest_idx:farthest_idx]
        if (self.traffic_waypoint_idx == -1) or (self.traffic_waypoint_idx >= farthest_idx):
            lane.waypoints = base_wps
        else:
            lane.waypoints = self.decel_waypoints(base_wps, closest_idx)

        return lane

    def decel_waypoints(self, base_wps, closest_idx):
        temp = []
        stop_idx = max(self.traffic_waypoint_idx - closest_idx - 4, 0) # Couple of waypoints before the stop line, so car stops correctly
        
        if (DEBUG_EN and (self.proc_count % LOG_FREQ == 0)):
            dist = self.distance(base_wps, 0, stop_idx)
            rospy.logwarn("CarIdx={}, StopLineIdx={}, StopIdx={}, Dist={}".format(closest_idx, self.traffic_waypoint_idx, stop_idx, dist))

        prev_vel = base_wps[0].twist.twist.linear.x
        for i, wp in enumerate(base_wps):

            p = Waypoint()
            p.pose = wp.pose
            
            dist = self.distance(base_wps, i, stop_idx)
            vel = math.sqrt(2 * TGT_DECEL * dist)
            # Limit velocity change to MAX_DECEL, if enough distance
            if (dist > MIN_DIST_DECEL_LIM):
                vel = max(vel, (prev_vel - MAX_DECEL))

            if (vel < 1.5):
                vel = 0.0
            if (DEBUG_EN and (self.proc_count % LOG_FREQ == 0)):
                rospy.logwarn("I={}, Vel={}, Dist={}".format(i, vel, dist))

            vel = min(vel, wp.twist.twist.linear.x)
            prev_vel = vel

            p.twist.twist.linear.x = vel
            
            temp.append(p)

        return temp
    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoint_tree:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message.
        self.traffic_waypoint_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
