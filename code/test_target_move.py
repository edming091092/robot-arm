"""Independent MG400 target test. Default mode only queries the controller."""
import argparse
import json
import math
from pathlib import Path

from mg400_tcp import MG400


def route(current, xyz, clearance_z):
    clearance = max(current[2], xyz[2], clearance_z)
    x, y, z = xyz
    r = current[3]
    points = [[*current[:2], clearance, r], [x, y, clearance, r], [x, y, z, r]]
    return [point for point, previous in zip(points, [current, *points]) if point != previous]


def check_route(bot, current, points):
    count = 0
    for start, end in zip([current, *points], points):
        steps = max(1, math.ceil(math.dist(start[:3], end[:3]) / 20))
        for index in range(1, steps + 1):
            pose = [a + (b - a) * index / steps for a, b in zip(start, end)]
            args = ",".join(f"{value:.4f}" for value in pose)
            result = bot.dashboard.request(f"InverseSolution({args},{bot.user},{bot.tool})")
            joints = [float(value) for value in result.split(",")]
            if len(joints) not in (4, 6) or not all(math.isfinite(value) for value in joints):
                raise RuntimeError(f"Unexpected inverse-kinematics result at {pose}: {result}")
            count += 1
    return count


def run_target_test(bot, xyz, clearance_z, execute=False):
    mode = bot.mode()
    bot.check_frames(bot.feedback.read(timeout=3, require_new=True))
    current = bot.pose()
    points = route(current, xyz, clearance_z)
    print(f"Current [X,Y,Z,R]: {current}")
    for number, point in enumerate(points, 1):
        print(f"Waypoint {number}: {point}")
    count = check_route(bot, current, points)
    print(f"Controller returned inverse-kinematics solutions for {count} sampled points.")
    if not execute:
        print("Read-only check complete. Physical path clearance has not been checked.")
        return
    if mode != 5 or bot.mode() != 5:
        raise RuntimeError("The robot must be enabled and idle for this test.")
    if not points:
        print("Already at the target.")
        return
    try:
        actual = bot.prepare()
        if (math.dist(actual[:3], current[:3]) > 0.5
                or abs((actual[3] - current[3] + 180) % 360 - 180) > 1):
            raise RuntimeError("Start position changed after the route check. Run the check again.")
        for point in points:
            bot.move(point)
        actual = bot.pose()
        if math.dist(actual[:3], xyz) > 1:
            raise RuntimeError(f"Target was not reached. Actual pose: {actual}")
        print(f"Target reached. Actual pose: {actual}")
    except BaseException:
        if bot.queue_owned:
            bot.stop()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("z", type=float)
    parser.add_argument("--clearance-z", type=float, required=True,
                        help="Measured collision-free travel height in mm")
    parser.add_argument("--execute", action="store_true",
                        help="Execute after checking the full physical route on site")
    args = parser.parse_args()
    xyz = [args.x, args.y, args.z]
    if not all(math.isfinite(value) for value in [*xyz, args.clearance_z]):
        parser.error("Coordinates must be finite")
    if args.clearance_z < args.z:
        parser.error("Clearance Z must be at or above the target Z")
    cfg_path = Path(__file__).resolve().parents[1] / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    bot = MG400(cfg["robot_ip"], cfg["user_index"], cfg["tool_index"], speed=3)
    try:
        run_target_test(bot, xyz, args.clearance_z, args.execute)
    finally:
        bot.close()


if __name__ == "__main__":
    main()
