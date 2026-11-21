# MG400 Random Motion - Rewritten from the English Manual

Run `code/random_move.py` in VS Code. **Direct execution controls the real robot.** The working folder and Python installation can be on any drive; the configuration is located relative to the script.

Required files:

```text
robot arm/
  config.json
  code/
    random_move.py
    mg400_tcp.py
```

This random program no longer imports `manual_move.py` or `robot.py`. Keep those files if you also use the manual direction program.

## Behavior

- User 0 workspace: X -50 to 50 mm, Y 200 to 300 mm, Z 0 to 50 mm.
- The initial pose may be outside the target workspace. The first straight-line MovL enters the box at the first random target; this entry segment may pass outside the box. The first target and confirmed arrival must be inside the box. Later moves keep the normal workspace checks. R remains at its initial angle.
- Ten random straight-line targets by default, with one second at each target.
- Global speed ratio: 5%; synchronized global SpeedL=100 and AccL=20 settings apply to each MovL. The manual states that global and command speed ratios multiply. This removes the previous double 5% setting (0.25% of the software playback speed).
- Enable, calibrated tool, payload, and TCP/IP mode must already be configured in the official software. No automatic EnableRobot or ClearError is used.

## Startup and completion

1. Read 30004 feedback and RobotMode. Require an enabled MG400, idle or paused, without alarms.
2. ResetRobot stops and clears the old queue. Only after its successful response, Continue releases the paused queue. No old move is intentionally resumed.
3. Set global User, Tool, SpeedL and AccL, then Sync before reading the starting pose. Confirm coordinate frame indexes from feedback. This controller rejects named GetPose parameters, so use GetPose() and the synchronized global frames. Its six-field response contains four useful MG400 pose fields followed by two extra fields; use the first four. MovL uses four positional arguments and the synchronized global settings.
4. Submit one MovL and check its acceptance response. Send Sync and monitor feedback while waiting.
5. Report pause, alarm, enable loss, feedback loss, movement timeout, or ten seconds without measured XYZ progress. Do not resend a motion command.
6. Confirm actual position after Sync before sending another target. On interruption or failure after taking ownership of the queue, attempt ResetRobot to clear remaining commands.

The program prints TX/RX responses, STATE, WAIT and DONE messages. Software stop commands cannot replace the physical emergency stop if communication or the controller fails.

## Optional terminal commands

Run these from the `robot arm` folder using your selected Python interpreter:

```powershell
python code/random_move.py --preview
python code/random_move.py --diagnose
python code/random_move.py --count 3 --speed 1
```

`--preview` does not connect. `--diagnose` only reads mode, alarms, feedback and pose; it does not clear or resume the queue. The third command performs real motion. `--execute` remains accepted for compatibility but is unnecessary.

## Manual basis

Source: `TCP_IP_4axis_EN.pdf`, 2024-04-19 edition, 68 PDF pages; its revision history corresponds to controller V1.6.0.0. Confirm the controller firmware if the command or feedback format is rejected.

| PDF pages | Used in the rewrite |
| --- | --- |
| 4-6 | Ports, ASCII framing, accepted versus completed commands |
| 10, 12-13 | ResetRobot, Pause, Continue and retained queues |
| 16, 49 | Multiplication of global and command speed ratios |
| 17-18, 30-31 | Coordinate frames, GetPose, RobotMode and GetErrorID |
| 51-52, 58 | MovL syntax and Sync completion |
| 63-67 | 1440-byte feedback, byte order, mode, pause, queue, enable and error flags |

The rewrite passed local TCP-simulator checks for fragmented replies, feedback validation, successful completion, pause, alarm, rejected commands without retry, stalled Sync, and Sync without actual arrival.

## Real robot validation - November 5, 2026

Following explicit user authorization, the corrected program completed one real target and then a complete run of 10 random real targets (seed 17) in the requested workspace. Every target returned Sync successfully and passed the position readback check. The program finished normally.

Observed controller compatibility differences:

- RunQueuedCmd=1 also occurs while enabled and idle. Do not treat it as a queue length or require it to be zero before moving. Use mode, pause and enable flags.
- GetPose(User=0,Tool=0) returns -30001 on this controller. GetPose() works after synchronized global frame selection.
- GetPose() returns six values; the first four match the MG400 X/Y/Z/R pose.
- CP(0) returns -40001 on this controller. It is omitted; each target is synchronized before another is submitted.

The complete 10-point run is recorded in `logs/real_motion_2026-11-05.txt`. Final readback was approximately X=38.2456, Y=263.8773, Z=15.4600 mm, R=227.0591 degrees. This validates the recorded run; coordinate readback does not establish calibrated physical accuracy or collision clearance for future runs.

The later change allowing an outside starting pose was verified offline only. It adds no automatic lift or obstacle avoidance; confirm clearance of the direct entry segment from the actual starting pose.
