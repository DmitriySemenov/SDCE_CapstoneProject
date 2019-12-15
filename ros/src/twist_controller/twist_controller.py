
from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
STOPPED_BRK_TRQ = 700 #Nm


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
    wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle, sample_freq):
        
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        
        kp_th = 0.3
        ki_th = 0.1
        kd_th = 0.0
        th_min = 0.0
        th_max = 0.4
        self.throttle_controller = PID(kp_th, ki_th, kd_th, th_min, th_max)

        tau = 0.5  # 1/(2pi*tau) = cutoff frequency
        ts = 1/float(sample_freq)  # Sample time
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0
        
        current_vel = self.vel_lpf.filt(current_vel)

        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0.0

        if linear_vel == 0.0 and current_vel < 0.1:
            throttle = 0.0
            brake = STOPPED_BRK_TRQ  # N*m - to hold the car in place if we are stopped at a light. Acceleration ~ 1m/s^2

        elif throttle < 0.1 and vel_error < 0:
            throttle = 0.0
            decel = max(vel_error, self.decel_limit)
            brake = min(STOPPED_BRK_TRQ, (abs(decel) * self.vehicle_mass * self.wheel_radius))  # Torque N*m

        return throttle, brake, steering