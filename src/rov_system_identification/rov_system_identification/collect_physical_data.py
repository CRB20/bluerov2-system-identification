#!/usr/bin/env python3
"""Extensive physical 6-DOF ROV identification-data collector.

The collector publishes generalized ControlForce directly to /rov/control_force.
Disable the normal velocity/position controllers before running it.

Protocols included:
  * individual: +/- pulses on all six axes at amplitudes 10..100 by 10.
  * cross: simultaneous multi-axis excitation; default Fx/Fy with four sign pairs.
  * broadband: PRBS, multisine and chirp blocks, each <= 5 s, separated by 5 s neutral.

At every amplitude level the operator is asked to return the ROV to the center
of the tank before the automatic tests continue.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required for the collector.") from exc

import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from rclpy.node import Node
from rov_msgs.msg import ControlForce, ThrusterPWM, VehicleState

from .common import INPUT_NAMES, ensure_dirs, environment_metadata, load_config, save_json


# Keep the diagnostic actuator mix consistent with the deployed allocator.
# This does NOT command PWM from the collector; it only records the expected
# pre-saturation thruster mix so saturation/dead-zone effects can be analysed.
SURGE = np.array([ 1,  1,  1,  1,  0,  0,  0,  0], dtype=float)
SWAY  = np.array([-1,  1,  1, -1,  0,  0,  0,  0], dtype=float)
HEAVE = np.array([ 0,  0,  0,  0, -1,  1,  1, -1], dtype=float)
ROLL  = np.array([ 0,  0,  0,  0, -1, -1,  1,  1], dtype=float)
PITCH = np.array([ 0,  0,  0,  0,  1, -1,  1, -1], dtype=float)
YAW   = np.array([-1,  1, -1,  1,  0,  0,  0,  0], dtype=float)
PWM_NEUTRAL = 1500.0
PWM_MIN = 1400.0
PWM_MAX = 1600.0


class PhysicalDataCollector(Node):
    def __init__(self, cfg: dict, session_root: Path):
        super().__init__("rov_system_id_collector")
        self.cfg = cfg
        self.root = session_root
        self.raw_dir = self.root / "data" / "raw"
        self.meta_dir = self.root / "metadata"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        c = cfg["collector"]
        self.sample_rate = float(c.get("sample_rate_hz", 50.0))
        self.pulse_seconds = min(float(c.get("pulse_seconds", 5.0)), 5.0)
        self.neutral_seconds = max(float(c.get("neutral_seconds", 5.0)), 5.0)
        self.settle_seconds = max(float(c.get("center_settle_seconds", 5.0)), 0.0)
        self.amplitudes = [float(a) for a in c.get("amplitudes", list(range(10, 101, 10)))]
        self.amplitudes = sorted(set(a for a in self.amplitudes if 0.0 < a <= 100.0))
        self.include_zero = bool(c.get("include_zero_baseline", True))
        self.repetitions = max(1, int(c.get("repetitions", 1)))
        self.do_individual = bool(c.get("individual", True))
        self.do_cross = bool(c.get("cross", True))
        self.do_broadband = bool(c.get("broadband", True))
        self.cross_pairs = [tuple(x) for x in c.get("cross_pairs", [["Fx", "Fy"]])]
        self.broadband_types = [str(x).lower() for x in c.get("broadband_types", ["prbs", "multisine", "chirp"])]
        self.broadband_amplitudes = [float(a) for a in c.get("broadband_amplitudes", [30, 60, 90])]
        self.broadband_block_seconds = min(float(c.get("broadband_block_seconds", 5.0)), 5.0)
        self.prbs_segment_seconds = max(float(c.get("prbs_segment_seconds", 0.5)), 0.05)
        self.chirp_f0 = float(c.get("chirp_f0_hz", 0.05))
        self.chirp_f1 = float(c.get("chirp_f1_hz", 1.0))
        self.multisine_freqs = [float(x) for x in c.get("multisine_frequencies_hz", [0.10, 0.20, 0.40, 0.80])]
        self.broadband_axes = [str(x) for x in c.get("broadband_axes", ["Fx", "Fy", "Fz", "Mz"])]
        self.require_pwm = bool(c.get("require_thruster_pwm", True))
        self.pwm_freshness_seconds = max(float(c.get("pwm_freshness_seconds", 0.5)), 0.05)

        if not self.amplitudes and not self.broadband_amplitudes:
            raise ValueError("No nonzero amplitudes configured.")
        if any(abs(a) > 100.0 for a in self.amplitudes + self.broadband_amplitudes):
            raise ValueError("All identification amplitudes must be <= 100.")
        for pair in self.cross_pairs:
            if len(pair) < 2 or any(x not in INPUT_NAMES for x in pair):
                raise ValueError(f"Invalid cross pair: {pair}")
        for axis in self.broadband_axes:
            if axis not in INPUT_NAMES:
                raise ValueError(f"Invalid broadband axis: {axis}")

        # Exact VehicleState field order from rov_msgs/msg/VehicleState:
        # [x,y,z,vx,vy,vz,roll,pitch,yaw,roll_rate,pitch_rate,yaw_rate]
        self.state = np.full(12, np.nan, dtype=float)
        self.state_received = False

        # Exact ThrusterPWM field order from rov_msgs/msg/ThrusterPWM.
        self.last_pwm = np.full(8, np.nan, dtype=float)
        self.pwm_received = False
        self.last_pwm_time = 0.0

        # Track the active allocator/control mode for contamination diagnostics.
        self.control_mode = "unknown"

        self.sample_counter = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.raw_dir / f"rov_identification_{self.session_id}.csv"
        self.metadata_path = self.meta_dir / f"rov_identification_{self.session_id}.json"
        self._csv_file = None
        self._writer = None
        self._prompt_event = threading.Event()
        self._prompt_thread = None
        self._prompt_answer = False
        self._prompt_choice = "quit"
        self._operator_prompt_active = False
        self._shutdown_requested = False

        self.sub = self.create_subscription(VehicleState, "/rov/state", self.state_callback, 50)
        self.force_pub = self.create_publisher(ControlForce, "/rov/control_force", 10)
        self.pwm_sub = self.create_subscription(ThrusterPWM, "/rov/thruster_pwm", self.pwm_callback, 50)
        self.mode_sub = self.create_subscription(String, "/rov/control_mode", self.mode_callback, 20)
        # Watch /rov/cmd_velocity only to detect accidental controller activity.
        self.cmd_sub = self.create_subscription(Twist, "/rov/cmd_velocity", self.velocity_watch, 10)
        self.last_velocity_cmd = np.zeros(6)

        self.csv_fields = [
            "session_id", "experiment_id", "segment_id", "protocol", "experiment_family",
            "amplitude", "repetition", "sequence_index", "pair_name", "sign_pattern",
            "phase", "phase_elapsed_s", "ros_time_s", "wall_time_iso", "sample_index",
            *[f"{n}_cmd" for n in INPUT_NAMES],
            "x", "y", "z", "vx", "vy", "vz",
            "roll", "pitch", "yaw", "roll_rate", "pitch_rate", "yaw_rate",
            "u", "v", "w", "p", "q", "r",
            *[f"thruster_mix_ch{i}" for i in range(1, 9)],
            *[f"pwm_ch{i}" for i in range(1, 9)],
            "pwm_received", "pwm_fresh", "pwm_age_s", "pwm_saturated_count", "pwm_min", "pwm_max",
            "control_mode", "velocity_command_detected", "operator_center_confirmed",
        ]
        self._open_csv()

    def _open_csv(self):
        import csv
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self.csv_fields)
        self._writer.writeheader()
        self._csv_file.flush()

    def state_callback(self, msg: VehicleState):
        # Exact VehicleState message mapping. Do not reorder these fields in the
        # stored CSV: x,y,z are pose; vx,vy,vz are linear velocity;
        # roll,pitch,yaw are attitude; roll_rate,pitch_rate,yaw_rate are rates.
        self.state[:] = [
            float(msg.x), float(msg.y), float(msg.z),
            float(msg.vx), float(msg.vy), float(msg.vz),
            float(msg.roll), float(msg.pitch), float(msg.yaw),
            float(msg.roll_rate), float(msg.pitch_rate), float(msg.yaw_rate),
        ]
        self.state_received = True

    def pwm_callback(self, msg: ThrusterPWM):
        # Exact ThrusterPWM message mapping.
        self.last_pwm[:] = [
            float(msg.ch1), float(msg.ch2), float(msg.ch3), float(msg.ch4),
            float(msg.ch5), float(msg.ch6), float(msg.ch7), float(msg.ch8),
        ]
        self.pwm_received = True
        self.last_pwm_time = time.monotonic()

    def mode_callback(self, msg: String):
        self.control_mode = msg.data.strip().lower() or "unknown"

    def velocity_watch(self, msg: Twist):
        self.last_velocity_cmd[:] = [
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z,
        ]

    def publish_force(self, tau: Sequence[float]):
        msg = ControlForce()
        msg.fx, msg.fy, msg.fz = [float(x) for x in tau[:3]]
        msg.mx, msg.my, msg.mz = [float(x) for x in tau[3:]]
        self.force_pub.publish(msg)

    def neutral(self):
        self.publish_force(np.zeros(6))

    def _write_row(self, experiment_id: str, segment_id: str, protocol: str, family: str,
                   amplitude: float, repetition: int, sequence_index: int,
                   pair_name: str, sign_pattern: str, phase: str,
                   phase_elapsed: float, tau: np.ndarray, operator_confirmed: bool):
        ros_time = self.get_clock().now().nanoseconds * 1e-9
        row = {
            "session_id": self.session_id,
            "experiment_id": experiment_id,
            "segment_id": segment_id,
            "protocol": protocol,
            "experiment_family": family,
            "amplitude": amplitude,
            "repetition": repetition,
            "sequence_index": sequence_index,
            "pair_name": pair_name,
            "sign_pattern": sign_pattern,
            "phase": phase,
            "phase_elapsed_s": phase_elapsed,
            "ros_time_s": ros_time,
            "wall_time_iso": datetime.now(timezone.utc).isoformat(),
            "sample_index": self.sample_counter,
            **{f"{n}_cmd": float(tau[i]) for i, n in enumerate(INPUT_NAMES)},
            "x": self.state[0], "y": self.state[1], "z": self.state[2],
            "vx": self.state[3], "vy": self.state[4], "vz": self.state[5],
            "roll": self.state[6], "pitch": self.state[7], "yaw": self.state[8],
            "roll_rate": self.state[9], "pitch_rate": self.state[10],
            "yaw_rate": self.state[11],
            "u": self.state[3], "v": self.state[4], "w": self.state[5],
            "p": self.state[9], "q": self.state[10], "r": self.state[11],
        }
        raw_mix = (
            float(tau[0]) * SURGE + float(tau[1]) * SWAY +
            float(tau[2]) * HEAVE + float(tau[3]) * ROLL +
            float(tau[4]) * PITCH + float(tau[5]) * YAW
        )
        pwm_ideal = np.clip(PWM_NEUTRAL + raw_mix, PWM_MIN, PWM_MAX)
        pwm_age = (time.monotonic() - self.last_pwm_time) if self.pwm_received else np.inf
        pwm_fresh = bool(self.pwm_received and pwm_age <= self.pwm_freshness_seconds)
        pwm_values = self.last_pwm if pwm_fresh else np.full(8, np.nan)
        finite_pwm = pwm_values[np.isfinite(pwm_values)]
        row.update({
            **{f"thruster_mix_ch{i+1}": float(raw_mix[i]) for i in range(8)},
            **{f"pwm_ch{i+1}": float(pwm_values[i]) for i in range(8)},
            "pwm_received": int(self.pwm_received),
            "pwm_fresh": int(pwm_fresh),
            "pwm_age_s": float(pwm_age) if np.isfinite(pwm_age) else np.nan,
            "pwm_saturated_count": int(np.sum((pwm_ideal <= PWM_MIN + 1e-9) | (pwm_ideal >= PWM_MAX - 1e-9))),
            "pwm_min": float(np.min(finite_pwm)) if finite_pwm.size else np.nan,
            "pwm_max": float(np.max(finite_pwm)) if finite_pwm.size else np.nan,
            "control_mode": self.control_mode,
            "velocity_command_detected": int(np.linalg.norm(self.last_velocity_cmd) > 1e-6),
            "operator_center_confirmed": int(operator_confirmed),
        })
        self._writer.writerow(row)
        self.sample_counter += 1

    def _spin_until(self, duration: float, command_fn, exp_id: str, segment_id: str,
                    protocol: str, family: str, amplitude: float, repetition: int,
                    sequence_index: int, pair_name: str, sign_pattern: str,
                    phase: str, operator_confirmed: bool):
        start = time.monotonic()
        while rclpy.ok() and not self._shutdown_requested:
            elapsed = time.monotonic() - start
            if elapsed >= duration:
                break
            tau = np.asarray(command_fn(elapsed), dtype=float)
            tau = np.clip(tau, -100.0, 100.0)
            self.publish_force(tau)
            rclpy.spin_once(self, timeout_sec=min(0.002, max(0.0005, 1.0 / self.sample_rate)))
            if self.state_received:
                self._write_row(
                    exp_id, segment_id, protocol, family, amplitude, repetition,
                    sequence_index, pair_name, sign_pattern, phase, elapsed, tau,
                    operator_confirmed,
                )
        self.neutral()

    def _neutral_segment(self, exp_id: str, family: str, protocol: str, amplitude: float,
                         repetition: int, sequence_index: int, pair_name: str, sign_pattern: str,
                         operator_confirmed: bool, duration: float | None = None):
        duration = self.neutral_seconds if duration is None else max(0.0, float(duration))
        segment_id = f"{exp_id}_neutral_{int(time.time()*1000)}"
        self._spin_until(
            duration,
            lambda _t: np.zeros(6),
            exp_id, segment_id, protocol, family, amplitude, repetition, sequence_index,
            pair_name, sign_pattern, "neutral", operator_confirmed,
        )

    def _pulse_segment(self, exp_id: str, family: str, amplitude: float, repetition: int,
                       sequence_index: int, tau: np.ndarray, sign_pattern: str,
                       pair_name: str, operator_confirmed: bool, protocol="individual"):
        segment_id = f"{exp_id}_pulse"
        self._spin_until(
            self.pulse_seconds,
            lambda _t, tau=tau: tau,
            exp_id, segment_id, protocol, family, amplitude, repetition, sequence_index,
            pair_name, sign_pattern, "pulse", operator_confirmed,
        )
        self._neutral_segment(exp_id, family, protocol, amplitude, repetition, sequence_index,
                              pair_name, sign_pattern, operator_confirmed)

    def _prompt_center(self, label: str) -> bool:
        """Keep publishing neutral and spinning while waiting for operator confirmation."""
        self._prompt_answer = False
        self._prompt_event.clear()
        self._operator_prompt_active = True

        def ask():
            try:
                ans = input(
                    f"\n[{label}] Bring the ROV to the MIDDLE OF THE TANK, clear of obstacles, "
                    "then press ENTER to continue (type q to quit): "
                ).strip().lower()
                self._prompt_answer = ans not in {"q", "quit", "x", "exit"}
            except EOFError:
                self._prompt_answer = False
            finally:
                self._prompt_event.set()

        self._prompt_thread = threading.Thread(target=ask, daemon=True)
        self._prompt_thread.start()
        while rclpy.ok() and not self._prompt_event.is_set() and not self._shutdown_requested:
            self.neutral()
            rclpy.spin_once(self, timeout_sec=0.05)
        self._operator_prompt_active = False
        if self._prompt_event.is_set():
            return self._prompt_answer
        return False

    def _prompt_amplitude_complete(self, amplitude: float) -> str:
        """After an amplitude block, let the operator continue, repeat, or quit.

        While waiting, the collector continues to command neutral force. The operator
        should physically return the ROV to the middle of the tank before choosing
        ENTER (continue) or R (repeat).
        """
        self._prompt_choice = "quit"
        self._prompt_event.clear()
        self._operator_prompt_active = True

        def ask():
            try:
                ans = input(
                    f"\n[AMPLITUDE {amplitude:g} COMPLETE] Bring the ROV to the MIDDLE OF THE TANK. "
                    "Press ENTER to continue, R to REPEAT THIS AMPLITUDE, or Q to quit: "
                ).strip().lower()
                if ans in {"r", "repeat"}:
                    self._prompt_choice = "repeat"
                elif ans in {"q", "quit", "x", "exit"}:
                    self._prompt_choice = "quit"
                else:
                    self._prompt_choice = "continue"
            except EOFError:
                self._prompt_choice = "quit"
            finally:
                self._prompt_event.set()

        self._prompt_thread = threading.Thread(target=ask, daemon=True)
        self._prompt_thread.start()
        while rclpy.ok() and not self._prompt_event.is_set() and not self._shutdown_requested:
            self.neutral()
            rclpy.spin_once(self, timeout_sec=0.05)
        self._operator_prompt_active = False
        if not self._prompt_event.is_set():
            return "quit"
        return self._prompt_choice

    def _rollback_csv(self, file_offset: int, sample_counter: int):
        """Discard all rows written since file_offset and restore the sample counter."""
        self._csv_file.flush()
        self._csv_file.seek(file_offset)
        self._csv_file.truncate()
        self._csv_file.seek(0, os.SEEK_END)
        self.sample_counter = sample_counter

    def _write_metadata(self, status: str):
        payload = {
            "status": status,
            "session_id": self.session_id,
            "csv_file": str(self.csv_path),
            "collector_config": self.cfg.get("collector", {}),
            "ros_topics": {
                "state": "/rov/state",
                "control_force": "/rov/control_force",
                "thruster_pwm": "/rov/thruster_pwm",
                "velocity_command_watch": "/rov/cmd_velocity",
                "control_mode": "/rov/control_mode",
            },
            "ros_messages": {
                "VehicleState": [
                    "x", "y", "z", "vx", "vy", "vz",
                    "roll", "pitch", "yaw", "roll_rate", "pitch_rate", "yaw_rate"
                ],
                "ControlForce": ["fx", "fy", "fz", "mx", "my", "mz"],
                "ThrusterPWM": ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"],
                "PositionGoal": ["x", "y", "z", "yaw"],
            },
            "state_fields": ["x","y","z","vx","vy","vz","roll","pitch","yaw","roll_rate","pitch_rate","yaw_rate"],
            "input_fields": INPUT_NAMES,
            "sample_fields": self.csv_fields,
            "notes": [
                "ControlForce is the commanded generalized force/torque, not independently measured thrust.",
                "ThrusterPWM is logged when the deployed allocator is running, allowing actuator saturation/dead-zone diagnostics.",
                "Disable position/velocity/adaptive controllers during collection; the collector owns /rov/control_force.",
                "The allocator/bridge should remain active so the actual PWM path can be recorded; require_thruster_pwm defaults to true.",
                "Each excitation segment is <= 5 s and has >= 5 s neutral between experiments.",
                "The stored VehicleState fields follow the exact rov_msgs VehicleState message definition.",
            ],
            "environment": environment_metadata(),
        }
        save_json(self.metadata_path, payload)

    def collect_baseline(self):
        if not self._prompt_center("ZERO BASELINE"):
            return False
        exp_id = f"baseline_0_{self.session_id}"
        self._neutral_segment(exp_id, "baseline", "baseline", 0.0, 1, 0, "", "0", True,
                              duration=max(self.pulse_seconds, 5.0))
        return True

    def collect_individual(self, amplitude: float, repetition: int, operator_confirmed: bool):
        trial = 0
        for axis_index, axis in enumerate(INPUT_NAMES):
            for sign in (1, -1):
                if not rclpy.ok() or self._shutdown_requested:
                    return
                trial += 1
                tau = np.zeros(6)
                tau[axis_index] = sign * amplitude
                exp_id = f"individual_{axis}_a{amplitude:g}_r{repetition}_s{sign:+d}_{self.session_id}"
                self.get_logger().info(f"INDIVIDUAL {axis} {sign:+d} A={amplitude:g}")
                self._pulse_segment(
                    exp_id, "individual", amplitude, repetition, trial, tau,
                    f"{sign:+d}", "", operator_confirmed, protocol="individual"
                )

    def collect_cross_pair(self, pair: Sequence[str], amplitude: float, repetition: int,
                           operator_confirmed: bool):
        indices = [INPUT_NAMES.index(x) for x in pair]
        trial = 0
        for signs in itertools.product((1, -1), repeat=len(indices)):
            trial += 1
            tau = np.zeros(6)
            for idx, s in zip(indices, signs):
                tau[idx] = s * amplitude
            pair_name = "+".join(pair)
            sign_pattern = ",".join(f"{s:+d}" for s in signs)
            safe_sign = "".join("p" if s > 0 else "m" for s in signs)
            exp_id = f"cross_{pair_name.replace('+','_')}_a{amplitude:g}_r{repetition}_{safe_sign}_{self.session_id}"
            self.get_logger().info(f"CROSS {pair_name} signs={sign_pattern} A={amplitude:g}")
            self._pulse_segment(
                exp_id, "cross", amplitude, repetition, trial, tau,
                sign_pattern, pair_name, operator_confirmed, protocol="cross"
            )

    def _broadband_signal(self, kind: str, amp: float, axis_index: int):
        duration = self.broadband_block_seconds
        if kind == "prbs":
            def fn(t):
                bit = int(t / self.prbs_segment_seconds)
                val = amp if bit % 2 == 0 else -amp
                tau = np.zeros(6); tau[axis_index] = val
                return tau
            return fn
        if kind == "multisine":
            freqs = self.multisine_freqs
            def fn(t):
                value = sum(math.sin(2 * math.pi * f * t) for f in freqs)
                scale = max(1.0, len(freqs))
                tau = np.zeros(6); tau[axis_index] = amp * value / scale
                return tau
            return fn
        if kind == "chirp":
            f0, f1 = self.chirp_f0, self.chirp_f1
            k = (f1 - f0) / max(duration, 1e-9)
            def fn(t):
                phase = 2 * math.pi * (f0 * t + 0.5 * k * t * t)
                tau = np.zeros(6); tau[axis_index] = amp * math.sin(phase)
                return tau
            return fn
        raise ValueError(f"Unknown broadband kind: {kind}")

    def collect_broadband(self, amplitude: float, repetition: int, operator_confirmed: bool):
        trial = 0
        for kind in self.broadband_types:
            for axis in self.broadband_axes:
                trial += 1
                axis_index = INPUT_NAMES.index(axis)
                exp_id = f"broadband_{kind}_{axis}_a{amplitude:g}_r{repetition}_{self.session_id}"
                segment_id = f"{exp_id}_block"
                self.get_logger().info(f"BROADBAND {kind} {axis} A={amplitude:g}")
                self._spin_until(
                    self.broadband_block_seconds,
                    self._broadband_signal(kind, amplitude, axis_index),
                    exp_id, segment_id, "broadband", "broadband", amplitude, repetition,
                    trial, "", f"{kind}:{axis}", "broadband", operator_confirmed,
                )
                self._neutral_segment(
                    exp_id, "broadband", "broadband", amplitude, repetition, trial,
                    "", f"{kind}:{axis}", operator_confirmed,
                )

    def run(self):
        print("\n" + "=" * 90)
        print("ROV SYSTEM IDENTIFICATION — PHYSICAL DATA COLLECTION")
        print("=" * 90)
        print(f"Session: {self.session_id}")
        print(f"CSV:     {self.csv_path}")
        print(f"Pulse:   {self.pulse_seconds:.1f} s (hard maximum 5 s)")
        print(f"Neutral: {self.neutral_seconds:.1f} s (minimum 5 s)")
        print(f"Amplitudes: {self.amplitudes}")
        print("\nIMPORTANT: Disable the normal position/velocity/adaptive controller stack before continuing.")
        print("This node publishes directly to /rov/control_force. Keep the emergency stop ready.\n")

        if not self.state_received:
            print("Waiting for /rov/state...")
            deadline = time.monotonic() + 10.0
            while rclpy.ok() and time.monotonic() < deadline and not self.state_received:
                self.neutral()
                rclpy.spin_once(self, timeout_sec=0.05)
            if not self.state_received:
                raise RuntimeError("No /rov/state received within 10 seconds.")

        if self.require_pwm:
            print("Waiting for /rov/thruster_pwm from the allocator...")
            deadline = time.monotonic() + 10.0
            while rclpy.ok() and time.monotonic() < deadline and (
                not self.pwm_received or (time.monotonic() - self.last_pwm_time) > self.pwm_freshness_seconds
            ):
                self.neutral()
                rclpy.spin_once(self, timeout_sec=0.05)
            if not self.pwm_received or (time.monotonic() - self.last_pwm_time) > self.pwm_freshness_seconds:
                raise RuntimeError(
                    "No fresh /rov/thruster_pwm received. Start the allocator and MAVLink bridge before identification."
                )

        if self.include_zero and not self.collect_baseline():
            return

        for amp in self.amplitudes:
            # The operator must centre the vehicle before each amplitude block.
            if not self._prompt_center(f"AMPLITUDE {amp:g}"):
                break

            while rclpy.ok() and not self._shutdown_requested:
                # Remember the exact CSV position and sample counter so the entire
                # amplitude block can be discarded if the operator requests a repeat.
                self._csv_file.flush()
                amp_file_offset = self._csv_file.tell()
                amp_sample_counter = self.sample_counter

                for rep in range(1, self.repetitions + 1):
                    if self.do_individual:
                        self.collect_individual(amp, rep, True)
                    if self.do_cross:
                        for pair in self.cross_pairs:
                            self.collect_cross_pair(pair, amp, rep, True)
                    if self.do_broadband and amp in self.broadband_amplitudes:
                        self.collect_broadband(amp, rep, True)

                self._csv_file.flush()
                choice = self._prompt_amplitude_complete(amp)

                if choice == "continue":
                    break

                if choice == "repeat":
                    self.get_logger().warn(
                        f"Discarding all recorded data for amplitude {amp:g} and repeating it."
                    )
                    self._rollback_csv(amp_file_offset, amp_sample_counter)
                    # The operator has already confirmed that the ROV is back in the
                    # middle at the completion prompt, so repeat this amplitude directly.
                    continue

                self._shutdown_requested = True
                break

            if self._shutdown_requested:
                break

        self.neutral()
        self._write_metadata("complete")
        self._csv_file.flush()
        print("\nCollection complete.")
        print(f"Raw data: {self.csv_path}")
        print(f"Metadata : {self.metadata_path}")

    def close(self, status="interrupted"):
        self._shutdown_requested = True
        try:
            self.neutral()
        except Exception:
            pass
        try:
            if self._csv_file is not None:
                self._csv_file.flush()
                self._csv_file.close()
        finally:
            self._write_metadata(status)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect extensive physical ROV system-identification data.")
    parser.add_argument(
        "--root",
        default="~/rov_system_id_ws/ROV_system_idetification",
        help="ROV_system_idetification root folder.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML configuration. Defaults to package config_identification.yaml or <root>/config/identification.yaml.",
    )
    args = parser.parse_args(argv)

    root = Path(os.path.expanduser(args.root)).resolve()
    if args.config:
        config_path = Path(os.path.expanduser(args.config)).resolve()
    else:
        local = Path(__file__).resolve().parents[1] / "config_identification.yaml"
        config_path = local if local.exists() else root / "config" / "identification.yaml"
    cfg = load_config(config_path)

    rclpy.init(args=None)
    node = PhysicalDataCollector(cfg, root)
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warn("Collection interrupted by user.")
        node.close("interrupted")
    except Exception as exc:
        node.get_logger().error(f"Collection failed: {exc}")
        node.close("failed")
        raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
