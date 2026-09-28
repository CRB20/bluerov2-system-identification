#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node

from rov_msgs.msg import ControlForce
from rov_msgs.msg import ThrusterPWM
from std_msgs.msg import String


SURGE = np.array([ 1,  1,  1,  1,  0,  0,  0,  0], dtype=float)
SWAY  = np.array([-1,  1,  1, -1,  0,  0,  0,  0], dtype=float)
HEAVE = np.array([ 0,  0,  0,  0, -1,  1,  1, -1], dtype=float)
ROLL  = np.array([ 0,  0,  0,  0, -1, -1,  1,  1], dtype=float)
PITCH = np.array([ 0,  0,  0,  0,  1, -1,  1, -1], dtype=float)
YAW   = np.array([-1,  1, -1,  1,  0,  0,  0,  0], dtype=float)

PWM_NEUTRAL = 1500.0
PWM_MIN = 1400.0
PWM_MAX = 1600.0

# NEUTRAL_LOW = 1490.0
# NEUTRAL_HIGH = 1510.0

# MIN_USABLE_NEGATIVE = 1480.0
# MIN_USABLE_POSITIVE = 1520.0
NEUTRAL_LOW = 1500.0
NEUTRAL_HIGH = 1500.0

MIN_USABLE_NEGATIVE = 1500.0
MIN_USABLE_POSITIVE = 1500.0


CTRL_TIMEOUT_S = 0.5


class ROVAllocator(Node):
    def __init__(self):
        super().__init__("rov_allocator")

        self.sub = self.create_subscription(
            ControlForce,
            "/rov/control_force",
            self.force_callback,
            10
        )

        self.pub = self.create_publisher(
            ThrusterPWM,
            "/rov/thruster_pwm",
            10
        )

        self.last_force = np.zeros(8, dtype=int)
        self.last_force_time = self.get_clock().now()

        #self.get_logger().info("ROV allocator started")

        self.mode = "hold"

        self.mode_sub = self.create_subscription(
            String,
            "/rov/control_mode",
            self.mode_callback,
            10
        )

    @staticmethod
    def _to_pwm_message(pwm):

        msg = ThrusterPWM()

        msg.ch1 = int(pwm[0])
        msg.ch2 = int(pwm[1])
        msg.ch3 = int(pwm[2])
        msg.ch4 = int(pwm[3])
        msg.ch5 = int(pwm[4])
        msg.ch6 = int(pwm[5])
        msg.ch7 = int(pwm[6])
        msg.ch8 = int(pwm[7])

        return msg


    def mode_callback(self, msg: String):

        new_mode = msg.data.strip().lower()

        if new_mode in ("track", "hold"):

            if new_mode != self.mode:

                self.mode = new_mode

                # self.get_logger().info(
                #     f"Allocator mode -> {self.mode.upper()}"
                # )


    def force_callback(self, msg: ControlForce):
        self.last_force_time = self.get_clock().now()

        fx = float(msg.fx)
        fy = float(msg.fy)
        fz = float(msg.fz)
        mx = float(msg.mx)
        my = float(msg.my)
        mz = float(msg.mz)

        cmd = (
            fx * SURGE +
            fy * SWAY +
            fz * HEAVE +
            mx * ROLL +
            my * PITCH +
            mz * YAW
        )

        #pwm = np.clip(PWM_NEUTRAL + cmd, PWM_MIN, PWM_MAX).astype(int)
        # -------------------------------------------------
        # Convert force command to PWM
        # -------------------------------------------------

        pwm = np.clip(
            PWM_NEUTRAL + cmd,
            PWM_MIN,
            PWM_MAX
        )

        # -------------------------------------------------
        # TRACK MODE
        # -------------------------------------------------
        # Apply actuator deadband/remapping:
        #
        # 1400-1449 -> unchanged
        # 1450-1489 -> 1450
        # 1490-1510 -> 1500
        # 1511-1550 -> 1550
        # 1551-1600 -> unchanged
        # -------------------------------------------------

        if self.mode == "track":

            neutral_mask = (
                (pwm >= NEUTRAL_LOW) &
                (pwm <= NEUTRAL_HIGH)
            )

            pwm[neutral_mask] = PWM_NEUTRAL

            negative_small = (
                (pwm >= MIN_USABLE_NEGATIVE) &
                (pwm < NEUTRAL_LOW)
            )

            pwm[negative_small] = MIN_USABLE_NEGATIVE

            positive_small = (
                (pwm > NEUTRAL_HIGH) &
                (pwm <= MIN_USABLE_POSITIVE)
            )

            pwm[positive_small] = MIN_USABLE_POSITIVE

        # -------------------------------------------------
        # HOLD MODE
        # -------------------------------------------------
        # No deadband remapping.
        # Use the actual force-derived PWM.
        #
        # Example:
        #   +21 force -> 1521 PWM
        #   -21 force -> 1479 PWM
        # -------------------------------------------------

        elif self.mode == "hold":

            pass

        pwm = pwm.astype(int)
        
        self.last_force = pwm

        out = self._to_pwm_message(pwm)
        self.pub.publish(out)

        # self.get_logger().info(
        #     f"fx={fx:+.1f} fy={fy:+.1f} fz={fz:+.1f} mz={mz:+.1f} "
        #     f"-> pwm={pwm.tolist()}"
        # )

    def watchdog_callback(self):
        dt = (self.get_clock().now() - self.last_force_time).nanoseconds * 1e-9
        if dt > CTRL_TIMEOUT_S:
            neutral = np.array([1500] * 8, dtype=int)
            self.pub.publish(self._to_pwm_message(neutral))

    def emergency_stop(self):
        #self.get_logger().warn("Allocator emergency stop")

        neutral = np.array([1500] * 8, dtype=int)
        self.pub.publish(self._to_pwm_message(neutral))

        #self.get_logger().info("Allocator safe")


def main(args=None):
    rclpy.init(args=args)

    node = ROVAllocator()
    node.watchdog_timer = node.create_timer(0.1, node.watchdog_callback)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Allocator error: {e}")
    finally:
        node.emergency_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()