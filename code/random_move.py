"""MG400 random XYZ motion. Direct execution controls the real robot.
Based on TCP_IP_4axis_EN.pdf (2024-04-19), sections 1-4.
Use --preview for offline targets; --diagnose only reads controller state.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path
from mg400_tcp import MG400

ROOT = Path(__file__).resolve().parents[1]
LIMITS = {"x": [-50, 50], "y": [200, 300], "z": [0, 50]}


def check_limits(pose, limits=LIMITS):
    if len(pose) != 4 or not all(type(v) in (int, float) and math.isfinite(v) for v in pose):
        raise ValueError(f"Invalid [X,Y,Z,R]: {pose}")
    if not isinstance(limits, dict):
        raise ValueError("workspace_mm must contain x, y and z limits")
    for axis, value in zip("xyz", pose):
        bounds = limits.get(axis)
        if (not isinstance(bounds, list) or len(bounds) != 2
                or not all(type(v) in (int, float) and math.isfinite(v) for v in bounds)
                or bounds[0] > bounds[1]):
            raise ValueError(f"Invalid workspace bounds: {axis}={bounds}")
        if not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"{axis}={value} outside allowed target range {bounds}")


def make_targets(count, angle, seed=None):
    rng = random.Random(seed)
    return [[*(round(rng.uniform(*LIMITS[a]), 4) for a in "xyz"), angle] for _ in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="Offline preview; no connection")
    mode.add_argument("--diagnose", action="store_true", help="Read-only state/alarms/feedback")
    mode.add_argument("--execute", action="store_true", help="Real motion (also the default)")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--speed", type=int, default=5, help="Global speed ratio, 1-5 percent")
    args = parser.parse_args()
    if not 1 <= args.count <= 1000:
        parser.error("--count must be 1-1000")
    if not math.isfinite(args.interval) or not 0 <= args.interval <= 60:
        parser.error("--interval must be 0-60 seconds")
    if not 1 <= args.speed <= 5:
        parser.error("--speed must be 1-5 percent")
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    if type(cfg.get("user_index")) is not int or cfg["user_index"] != 0:
        parser.error("This workspace requires user_index=0")
    if type(cfg.get("tool_index")) is not int or not 0 <= cfg["tool_index"] <= 9:
        parser.error("tool_index must be a calibrated index from 0 to 9")

    def check(pose):
        check_limits(pose)
        check_limits(pose, cfg.get("workspace_mm"))

    bot = None
    print("XYZ range (mm): X [-50,50], Y [200,300], Z [0,50]; R held constant.", flush=True)
    try:
        if args.preview:
            start = [50, 250, 25, 10]
            print("PREVIEW ONLY: no connection; R=0 is illustrative.")
        else:
            bot = MG400(cfg["robot_ip"], cfg["user_index"], cfg["tool_index"], args.speed)
            if args.diagnose:
                bot.diagnose()
                bot.feedback.read(timeout=3, require_new=True)
                bot.status()
                print("Pose:", bot.pose())
                return
            print("REAL MOTION: clearing old queue, then executing 10 targets by default.", flush=True)
            start = bot.prepare()
            print("Start [X,Y,Z,R]:", start, flush=True)
            try:
                check(start)
            except ValueError:
                print("Start is outside the target range. The first straight-line move will enter it.", flush=True)
        targets = make_targets(args.count, start[3], args.seed)
        for i, target in enumerate(targets, 1):
            check(target)
            print(f"{i:03d}: {target}", flush=True)
        if args.preview:
            print("Preview complete. Running without --preview controls the real robot.")
            return
        print(f"Global speed {args.speed}%, MovL SpeedL=100; no second 5% multiplier.", flush=True)
        for i, target in enumerate(targets, 1):
            print(f"Moving {i}/{args.count}: {target}", flush=True)
            actual = bot.pose()
            if abs((actual[3] - start[3] + 180) % 360 - 180) > 1:
                raise RuntimeError("R changed unexpectedly; stopping")
            bot.move(target, check_pose=check, allow_start_outside=(i == 1))
            time.sleep(args.interval)
        print("Completed.", flush=True)
    except (KeyboardInterrupt, EOFError):
        print("Interrupted; stopping and clearing the motion queue.", flush=True)
        if bot and bot.queue_owned:
            bot.stop()
        raise SystemExit(130)
    except Exception as exc:
        print(f"STOPPED: {exc}", flush=True)
        if bot:
            if bot.queue_owned:
                bot.stop()
            bot.diagnose()
        raise SystemExit(1)
    finally:
        if bot:
            bot.close()


if __name__ == "__main__":
    main()
