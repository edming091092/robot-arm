"""MG400 four-axis protocol: English guide 2024-04-19, pp.4-6,10-18,30-31,49-67.
Dashboard 29999; motion 30003; little-endian 1440-byte feedback 30004.
Never automatically enable, clear alarms, or retry a movement.
"""
import math
import re
import select
import socket
import struct
import time


class Channel:
    def __init__(self, ip, port):
        self.ip, self.port = ip, port
        self.sock = None
        self.buffer = b""
        self.pending = None

    def begin(self, command):
        if self.pending is not None:
            raise RuntimeError(f"Port {self.port}: previous command still pending")
        if self.sock is None:
            self.sock = socket.create_connection((self.ip, self.port), timeout=3)
            self.sock.settimeout(3)
        print(f"[TX {self.port}] {command}", flush=True)
        self.pending = command
        try:
            self.sock.sendall(command.encode("ascii"))
        except OSError:
            self.close()
            raise

    def receive(self, timeout):
        """Return one complete response or None. Never resend a command."""
        if self.pending is None:
            raise RuntimeError("No response expected")
        if b";" not in self.buffer:
            if not select.select([self.sock], [], [], max(0, timeout))[0]:
                return None
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeError(f"Port {self.port}: controller disconnected")
            self.buffer += chunk
        if len(self.buffer) > 65536:
            raise RuntimeError("Controller response too long")
        if b";" not in self.buffer:
            return None
        raw, self.buffer = self.buffer.split(b";", 1)
        reply = raw.decode("ascii").strip() + ";"
        print(f"[RX {self.port}] {reply}", flush=True)
        command, self.pending = self.pending, None
        match = re.fullmatch(r"(-?\d+)\s*,\s*\{(.*)\}\s*,\s*(\w+)\(.*\);", reply, re.S)
        if not match or match[3].lower() != command.split("(", 1)[0].lower():
            raise RuntimeError(f"Unexpected response to {command}: {reply}")
        if int(match[1]) != 0:
            raise RuntimeError(f"Controller rejected {command}: {reply}")
        return match[2]

    def request(self, command, timeout=3):
        try:
            self.begin(command)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                result = self.receive(min(0.2, max(0, deadline - time.monotonic())))
                if result is not None:
                    return result
            raise TimeoutError(f"Port {self.port}: no complete response to {command}; not retried")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.sock:
            self.sock.close()
        self.sock = None
        self.buffer = b""
        self.pending = None


class Feedback:
    SIZE = 1440
    TEST_VALUE = 0x0123456789ABCDEF

    def __init__(self, ip, port=30004):
        self.ip, self.port = ip, port
        self.sock = None
        self.buffer = b""
        self.latest = None
        self.received_at = 0

    @staticmethod
    def decode(packet):
        if len(packet) != Feedback.SIZE or struct.unpack_from("<H", packet)[0] != Feedback.SIZE:
            raise ValueError("Invalid feedback packet length")
        if struct.unpack_from("<Q", packet, 48)[0] != Feedback.TEST_VALUE:
            raise ValueError("Feedback endian/format test failed; verify firmware compatibility")
        return {
            "mode": struct.unpack_from("<Q", packet, 24)[0],
            "speed_scaling": struct.unpack_from("<d", packet, 64)[0],
            "xyz": list(struct.unpack_from("<3d", packet, 624)),
            "user": packet[1012], "tool": packet[1013],
            "queued": packet[1014], "paused": packet[1015],
            "enabled": packet[1026], "error": packet[1029],
            "robot_type": packet[1031],
        }

    def read(self, timeout=0.1, require_new=False):
        if self.sock is None:
            self.sock = socket.create_connection((self.ip, self.port), timeout=3)
        deadline = time.monotonic() + timeout
        got_new = False
        while True:
            wait = 0 if got_new else max(0, deadline - time.monotonic())
            if not select.select([self.sock], [], [], wait)[0]:
                break
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeError("Feedback port disconnected")
            self.buffer += chunk
            while len(self.buffer) >= self.SIZE:
                packet, self.buffer = self.buffer[:self.SIZE], self.buffer[self.SIZE:]
                self.latest = self.decode(packet)
                self.received_at = time.monotonic()
                got_new = True
            if time.monotonic() >= deadline:
                break
        if (require_new and not got_new) or self.latest is None:
            raise TimeoutError("No new 1440-byte feedback packet; verify port 30004 and firmware")
        if time.monotonic() - self.received_at > 2:
            raise TimeoutError("Feedback is stale; stopping further motion")
        return self.latest

    def close(self):
        if self.sock:
            self.sock.close()
        self.sock = None


class MG400:
    def __init__(self, ip, user, tool, speed=5, ports=(29999, 30003, 30004)):
        self.dashboard = Channel(ip, ports[0])
        self.motion = Channel(ip, ports[1])
        self.feedback = Feedback(ip, ports[2])
        self.user, self.tool, self.speed = user, tool, speed
        self.queue_owned = False

    def mode(self):
        return int(self.dashboard.request("RobotMode()"))

    def pose(self):
        # This controller rejects named GetPose parameters (-30001). Read the
        # global frames, which prepare() sets and synchronizes before motion.
        result = self.dashboard.request("GetPose()")
        pose = [float(v) for v in result.split(",")]
        # The installed MG400 firmware returns four useful fields plus two
        # trailing fields. RobotType=1 is checked independently in prepare().
        if len(pose) not in (4, 6) or not all(math.isfinite(v) for v in pose):
            raise RuntimeError(f"Invalid GetPose response: {result}")
        return pose[:4]

    def status(self):
        state = self.feedback.read()
        print(f"[STATE] mode={state['mode']} enabled={state['enabled']} "
              f"queued={state['queued']} paused={state['paused']} error={state['error']}", flush=True)
        return state

    def prepare(self):
        self.feedback.read(timeout=3, require_new=True)
        state = self.status()
        mode = self.mode()
        if state["robot_type"] != 1:
            raise RuntimeError(f"Expected MG400 RobotType=1; got {state['robot_type']}")
        if mode not in (5, 10) or not state["enabled"] or state["error"]:
            raise RuntimeError(f"Robot must be enabled and idle/paused; RobotMode={mode}. "
                               "Check DobotStudio Pro, TCP/IP mode and alarms")
        # Never Continue a paused old queue. ResetRobot first stops/clears it.
        self.queue_owned = True
        self.dashboard.request("ResetRobot()")
        self.dashboard.request("Continue()")
        deadline = time.monotonic() + 5
        while True:
            state = self.feedback.read(timeout=0.2, require_new=True)
            if state["error"] or not state["enabled"]:
                raise RuntimeError("Robot lost enable or entered alarm during preparation")
            # RunQueuedCmd=1 can mean the queue runner is enabled while idle.
            # It is not a queue length or a reliable motion-busy flag.
            if not state["paused"] and state["mode"] == 5:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Queue remains paused/busy after ResetRobot and Continue")
        self.dashboard.request(f"SpeedFactor({self.speed})")
        # This firmware rejects CP(0). Each move is followed by Sync before
        # another target is submitted, so no blending parameter is required.
        for command in (f"User({self.user})", f"Tool({self.tool})", "SpeedL(100)", "AccL(20)"):
            self.dashboard.request(command)
        # Settings above are queued. Synchronize before GetPose, then confirm
        # that feedback reports the expected global coordinate indexes.
        self.motion.request("Sync()", timeout=5)
        state = self.feedback.read(timeout=0.2, require_new=True)
        self.check_frames(state)
        return self.pose()

    def check_frames(self, state):
        if state["user"] != self.user or state["tool"] != self.tool:
            raise RuntimeError(f"Coordinate frames changed: User={state['user']}, Tool={state['tool']}")

    def move(self, target, timeout=120, check_pose=None, allow_start_outside=False):
        current = self.pose()
        if check_pose:
            check_pose(target)
            if not allow_start_outside:
                check_pose(current)
        if math.dist(current[:3], target[:3]) <= 0.05 and not allow_start_outside:
            print("[SKIP] XYZ target already reached", flush=True)
            return
        state = self.feedback.read(timeout=0.2, require_new=True)
        self.check_frames(state)
        if state["mode"] != 5 or state["paused"] or state["error"] or not state["enabled"]:
            raise RuntimeError(f"Robot is not ready for a new move: {state}")
        command = "MovL(" + ",".join(f"{v:.4f}" for v in target) + ")"
        self.motion.request(command)
        self.motion.begin("Sync()")
        started = last_report = time.monotonic()
        last_progress = started
        progress_xyz = current[:3]
        while True:
            state = self.feedback.read(timeout=0.1)
            self.check_frames(state)
            if state["error"] or state["mode"] == 9:
                raise RuntimeError("Alarm during motion: " + self.dashboard.request("GetErrorID()"))
            if state["paused"] or state["mode"] == 10:
                raise RuntimeError("Motion queue paused; no automatic resume during motion")
            if not state["enabled"] or state["mode"] not in (5, 7):
                raise RuntimeError(f"Unexpected robot state during motion: {state}")
            result = self.motion.receive(0)
            if result is not None:
                actual = self.pose()
                if check_pose:
                    check_pose(actual)
                angle_error = abs((actual[3] - target[3] + 180) % 360 - 180)
                if math.dist(actual[:3], target[:3]) > 1 or angle_error > 1:
                    raise RuntimeError(f"Sync returned but target was not reached: {actual}")
                print(f"[DONE] {actual}", flush=True)
                return
            now = time.monotonic()
            if now - last_report >= 1:
                actual = self.pose()
                # The first entry segment may pass outside the target box.
                # Arrival is still checked above; later moves stay restricted.
                if check_pose and not allow_start_outside:
                    check_pose(actual)
                print(f"[WAIT] mode={state['mode']} queued={state['queued']} "
                      f"paused={state['paused']} pose={actual}", flush=True)
                if math.dist(actual[:3], progress_xyz) > 0.05:
                    last_progress = now
                    progress_xyz = actual[:3]
                if now - last_progress > 10:
                    raise TimeoutError("No measured XYZ progress for 10 seconds while awaiting Sync")
                last_report = now
            if now - started > timeout:
                raise TimeoutError(f"Motion did not complete within {timeout} seconds")

    def stop(self):
        try:
            self.dashboard.request("ResetRobot()")
            print("[STOP] ResetRobot acknowledged; planned queue cleared", flush=True)
        except Exception as exc:
            print(f"[STOP FAILED] {exc}. Use the physical emergency stop.", flush=True)

    def diagnose(self):
        for command in ("RobotMode()", "GetErrorID()"):
            try:
                self.dashboard.request(command)
            except Exception as exc:
                print(f"[DIAG] {exc}", flush=True)

    def close(self):
        for channel in (self.dashboard, self.motion, self.feedback):
            channel.close()
