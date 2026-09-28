#!/usr/bin/env python3

import time
import numpy as np

from pymavlink import mavutil

import rclpy
from rclpy.node import Node

from rov_msgs.msg import VehicleState
from rov_msgs.msg import ThrusterPWM


PWM_NEUTRAL = 1500
PWM_MIN = 1400
PWM_MAX = 1600

MAVLINK_RX_TIMEOUT_S = 0.5
STATE_TIMEOUT_S = 0.5
PWM_TIMEOUT_S = 0.5
STARTUP_HEARTBEAT_TIMEOUT_S = 5.0

STATE_FILTER_ALPHA = 0.8 #Filtering coefficient for state messages. 

ARM_ON_START = True


class MavlinkBridge(Node):
    def __init__(self):
        super().__init__("mavlink_bridge")

        self.state_pub = self.create_publisher(
            VehicleState,
            "/rov/state",
            10
        )

        self.pwm_sub = self.create_subscription(
            ThrusterPWM,
            "/rov/thruster_pwm",
            self.thruster_callback,
            10
        )

        self.get_logger().info("Connecting to MAVLink...")

        self.master = mavutil.mavlink_connection(
            "udpin:0.0.0.0:14550"
        )

        if not self._wait_for_heartbeat():
            raise RuntimeError("Could not receive MAVLink heartbeat")

        self.get_logger().info("Heartbeat received")

        # Latest vehicle state
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0

        # Filtered state
        self.x_f = 0.0
        self.y_f = 0.0
        self.z_f = 0.0

        self.vx_f = 0.0
        self.vy_f = 0.0
        self.vz_f = 0.0

        self.roll_f = 0.0
        self.pitch_f = 0.0
        self.yaw_f = 0.0

        self.roll_rate_f = 0.0
        self.pitch_rate_f = 0.0
        self.yaw_rate_f = 0.0

        self.filter_initialized = False

        # Latest thruster command
        self.last_pwm = [PWM_NEUTRAL] * 8

        # Timing / safety
        now = self.get_clock().now()
        self.last_mavlink_rx_time = now
        self.last_state_time = now
        self.last_pwm_time = now

        self.state_timeout_warned = False
        self.pwm_timeout_warned = False
        self.mav_timeout_warned = False

        if ARM_ON_START:
            self.get_logger().info("Sending neutral...")
            self.send_neutral()

            self.get_logger().info("Arming vehicle...")
            try:
                self.master.arducopter_arm()
                time.sleep(2.0)
                self.get_logger().info("Vehicle armed")
            except Exception as e:
                self.get_logger().error(f"Arming failed: {e}")
                raise

        self.timer = self.create_timer(0.05, self.update)

        self.get_logger().info("MAVLink bridge started")

    def _wait_for_heartbeat(self) -> bool:
        start = time.time()

        while True:
            hb = self.master.wait_heartbeat(timeout=1.0)
            if hb is not None:
                return True

            elapsed = time.time() - start
            self.get_logger().warn("Waiting for heartbeat...")

            if elapsed > STARTUP_HEARTBEAT_TIMEOUT_S:
                return False

    

    def thruster_callback(self, msg: ThrusterPWM):

        self.last_pwm = [
            int(msg.ch1),
            int(msg.ch2),
            int(msg.ch3),
            int(msg.ch4),
            int(msg.ch5),
            int(msg.ch6),
            int(msg.ch7),
            int(msg.ch8),
        ]

        self.last_pwm_time = self.get_clock().now()
        self.pwm_timeout_warned = False


    def send_pwm(self, pwm):
        pwm = [int(max(PWM_MIN, min(PWM_MAX, v))) for v in pwm]

        self.master.mav.rc_channels_override_send(
            self.master.target_system,
            self.master.target_component,
            *pwm
        )

    def send_neutral(self):
        self.send_pwm([PWM_NEUTRAL] * 8)

    def _mavlink_receive(self):
        got_any = False

        while True:
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                break

            got_any = True
            self.last_mavlink_rx_time = self.get_clock().now()
            self.mav_timeout_warned = False

            mtype = msg.get_type()

            if mtype == "LOCAL_POSITION_NED":
                self.x = msg.x
                self.y = msg.y
                self.z = msg.z

                self.vx = msg.vx
                self.vy = msg.vy
                self.vz = msg.vz

                self.last_state_time = self.get_clock().now()
                self.state_timeout_warned = False

            elif mtype == "ATTITUDE":
                self.roll = msg.roll
                self.pitch = msg.pitch
                self.yaw = msg.yaw

                self.roll_rate = msg.rollspeed
                self.pitch_rate = msg.pitchspeed
                self.yaw_rate = msg.yawspeed

                self.last_state_time = self.get_clock().now()
                self.state_timeout_warned = False

        return got_any


    def _wrap_pi(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle


    def _lowpass(self, previous, current):
        alpha = STATE_FILTER_ALPHA

        return (
            alpha * previous
            + (1.0 - alpha) * current
        )


    def _lowpass_angle(self, previous, current):
        alpha = STATE_FILTER_ALPHA

        error = self._wrap_pi(current - previous)

        return self._wrap_pi(
            previous + (1.0 - alpha) * error
        )


    def _filter_state(self):

        if not self.filter_initialized:

            self.x_f = self.x
            self.y_f = self.y
            self.z_f = self.z

            self.vx_f = self.vx
            self.vy_f = self.vy
            self.vz_f = self.vz

            self.roll_f = self.roll
            self.pitch_f = self.pitch
            self.yaw_f = self.yaw

            self.roll_rate_f = self.roll_rate
            self.pitch_rate_f = self.pitch_rate
            self.yaw_rate_f = self.yaw_rate

            self.filter_initialized = True

            return

        # Position
        self.x_f = self._lowpass(self.x_f, self.x)
        self.y_f = self._lowpass(self.y_f, self.y)
        self.z_f = self._lowpass(self.z_f, self.z)

        # Linear velocity
        self.vx_f = self._lowpass(self.vx_f, self.vx)
        self.vy_f = self._lowpass(self.vy_f, self.vy)
        self.vz_f = self._lowpass(self.vz_f, self.vz)

        # Roll / pitch
        self.roll_f = self._lowpass(self.roll_f, self.roll)
        self.pitch_f = self._lowpass(self.pitch_f, self.pitch)

        # Yaw: angle-aware filtering
        self.yaw_f = self._lowpass_angle(
            self.yaw_f,
            self.yaw
        )

        # Angular rates
        self.roll_rate_f = self._lowpass(
            self.roll_rate_f,
            self.roll_rate
        )

        self.pitch_rate_f = self._lowpass(
            self.pitch_rate_f,
            self.pitch_rate
        )

        self.yaw_rate_f = self._lowpass(
            self.yaw_rate_f,
            self.yaw_rate
        )


    def _publish_state(self):

        self._filter_state()

        state = VehicleState()

        state.x = float(self.x_f)
        state.y = float(self.y_f)
        state.z = float(self.z_f)

        state.vx = float(self.vx_f)
        state.vy = float(self.vy_f)
        state.vz = float(self.vz_f)

        state.roll = float(self.roll_f)
        state.pitch = float(self.pitch_f)
        state.yaw = float(self.yaw_f)

        state.roll_rate = float(self.roll_rate_f)
        state.pitch_rate = float(self.pitch_rate_f)
        state.yaw_rate = float(self.yaw_rate_f)

        self.state_pub.publish(state)

    def _freshness_checks(self):
        now = self.get_clock().now()

        mav_dt = (now - self.last_mavlink_rx_time).nanoseconds * 1e-9
        state_dt = (now - self.last_state_time).nanoseconds * 1e-9
        pwm_dt = (now - self.last_pwm_time).nanoseconds * 1e-9

        mav_ok = mav_dt <= MAVLINK_RX_TIMEOUT_S
        state_ok = state_dt <= STATE_TIMEOUT_S
        pwm_ok = pwm_dt <= PWM_TIMEOUT_S

        return mav_ok, state_ok, pwm_ok, mav_dt, state_dt, pwm_dt

    def update(self):
        self._mavlink_receive()
        self._publish_state()

        mav_ok, state_ok, pwm_ok, mav_dt, state_dt, pwm_dt = self._freshness_checks()

        #If MAVLink itself goes quiet, force neutral.
        if not mav_ok:
            if not self.mav_timeout_warned:
                self.get_logger().error(
                    f"MAVLink timeout ({mav_dt:.2f}s). Sending neutral."
                )
                self.mav_timeout_warned = True

            self.send_neutral()
            return

        # If vehicle state is stale, do not trust control output.
        if not state_ok:
            if not self.state_timeout_warned:
                self.get_logger().error(
                    f"State timeout ({state_dt:.2f}s). Sending neutral."
                )
                self.state_timeout_warned = True

            self.send_neutral()
            return

        # If no PWM commands are arriving, also go neutral.
        if not pwm_ok:
            if not self.pwm_timeout_warned:
                self.get_logger().warn(
                    f"PWM timeout ({pwm_dt:.2f}s). Sending neutral."
                )
                self.pwm_timeout_warned = True

            self.send_neutral()
            return

        # Normal operation
        try:
            self.send_pwm(self.last_pwm)
        except Exception as e:
            self.get_logger().error(f"Failed to send PWM: {e}")
            self.send_neutral()

    def emergency_stop(self):
        self.get_logger().warn("Emergency stop")

        try:
            self.send_neutral()
        except Exception:
            pass

        for _ in range(50):
            try:
                self.send_neutral()
            except Exception:
                pass
            time.sleep(0.1)

        try:
            self.master.arducopter_disarm()
        except Exception:
            pass

        self.get_logger().info("Vehicle safe")


def main(args=None):
    rclpy.init(args=args)

    node = None
    try:
        node = MavlinkBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node is not None:
            node.get_logger().error(f"Bridge error: {e}")
        else:
            print(f"Bridge error: {e}")
    finally:
        if node is not None:
            node.emergency_stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()