# deadcat-tracker
A little tracking drone project.

## Running the tracker

Run commands from this repository's root. The default is a camera dry run:

```sh
.venv/bin/python main.py
```

Connect the OAK-D Lite over USB. The program prints tracking status, calculated
forward velocity/yaw rate, observation FPS, frame age, and the last target position
about once per second; it does not open the UART in this mode. Position and yaw
errors remain visible even when navigation gains are zero. Set the desired
following distance in meters with `--standoff` (default
`2.0`):

```sh
.venv/bin/python main.py --standoff 2.5
```

Camera startup needs the DepthAI dependencies, a working USB connection, adequate
power, and access to the YOLO model archive (cached locally or downloaded on first
use). On the Pi, check the Linux USB permissions described in the
[Luxonis USB deployment guide](https://docs.luxonis.com/hardware/platform/deploy/usb-deployment-guide/).
Camera errors are reported by the program; the camera pipeline still needs
verification on the actual OAK-D Lite/Pi.

### Checking detection geometry and performance on the Pi

The pipeline loads the full model archive into the detection parser, including
its input tensor dimensions (512 wide by 288 high for the current model). It
requests an undistorted inference image at that size and uses the neural network's
passthrough image as `StereoDepth.inputAlignTo`. This is the RVC2 alignment path;
there is no independent 640x400 depth-output resize. This follows the approach in
[Luxonis's spatial detection implementation](https://github.com/luxonis/depthai-core/blob/main/src/pipeline/node/SpatialDetectionNetwork.cpp).
The first depth/detection transformations must match each other and the model
dimensions before the program accepts observations. Warnings are not suppressed.

Run without snapshots first to measure the normal camera-processing load:

```sh
.venv/bin/python main.py
```

Look for `Model input=512x288` followed by
`Verified depth/detection geometry: 512x288`. The repeated 416x416 fallback and
transformation-remapping warnings should be gone. If the geometry check fails,
the program stops and reports it instead of passing those coordinates to control.

- `vision_fps`: accepted spatial observations per second, including frames with
  no target. This is independent of the 10 Hz control loop; the first interval
  shows zero while the measurement starts.
- `age_ms`: age of the most recent observation at logging time. It should remain
  comfortably below the existing 500 ms expiry limit. A stopped stream makes age
  grow and FPS fall to zero.
- `last_target_xyz_m`: forward, right and down coordinates in metres. Compare
  the first number with a measured forward distance from the camera while the
  person is centred and stationary. These are the last observation's values;
  a stale status means they must not be treated as current.
- `yaw_error_deg`: target angle to the right (positive) or left (negative).

To inspect bounding boxes on a Pi without a desktop display, optionally run:

```sh
.venv/bin/python main.py --snapshot-dir camera-check
```

This saves up to 30 annotated JPEGs per run, at most one per second, with unique
names. Open or copy them from `camera-check`. Green marks the tracked person;
yellow marks other person detections, including candidates when tracking is lost.
The `z` label is forward depth in metres. Images are matched to detections by both
sequence number and capture timestamp; unmatched images are skipped. Check boxes
with the person near the centre and near the image edges. Restart after prolonged
target loss as described below. Snapshot capture uses the existing headless OpenCV
package and is disabled in live mode. It adds image transfer and disk-write work,
so compare performance using the run without snapshots.

No camera measurements or before/after timing results from the Pi are bundled
with this change; the checks above are the remaining hardware validation.

After configuring the dependencies, UART connection, flight controller, and control
gains, live transmission is explicitly enabled with:

```sh
.venv/bin/python main.py --live
```

Live mode uses the existing `mavlink_interface.py` connection settings:
`/dev/serial0`, `57600` baud, and source system ID `245`. The planned connection is
the Pi UART to the Pixhawk 6C Mini's TELEM1 port. Port setup, matching MAVLink/baud
settings, dependencies, and gain tuning remain separate setup work. Both gains in
`navigation.py` are still `0.0`, so calculated movement commands currently remain
zero even with a tracked target.

## Enabling and stopping tracking with the transmitter

The live controller expects **ArduCopter**. It observes the flight controller's
armed state and GUIDED mode; it never arms, takes off, or changes flight modes.

Use a spare transmitter switch configured on the flight controller as
`RCx_OPTION = 55` (GUIDED), where `x` is the actual spare RC input channel. That
channel must not also control a flight axis or the normal flight-mode selection.
Keep the normal mode switch in a suitable pilot-controlled mode, such as Loiter,
so lowering the auxiliary switch can return to that mode. A normal flight-mode
switch with a GUIDED position is also supported; the program observes the resulting
flight mode rather than reading a particular RC channel. These are configuration
instructions for later; the program does not write RC parameters.
[ArduPilot auxiliary mode switches](https://ardupilot.org/copter/docs/common-auxiliary-functions.html#mode-switches)

1. Start the program with the aircraft outside GUIDED and wait for the program to
   observe that state. Starting already in GUIDED does not enable tracking.
2. Arm and take off manually in a suitable flight mode. Acquire the intended
   person in the camera view.
3. Switch into GUIDED to enable tracking. The program must observe both GUIDED and
   the armed state before it can transmit following commands.
4. Switch out of GUIDED to take control. The program stops sending setpoints once
   it observes the mode change. Disarming also disables transmission.

An auxiliary mode switch returns to the normal mode selection when lowered only
if the aircraft is still in the mode selected by that auxiliary switch. Mode
changes can be denied by ArduCopter, so verify the actual mode. To reset a tracking
fault, leave GUIDED, wait for the program to report the disabled/ready state, then
switch back into GUIDED; a quick out-and-back toggle can be missed between
heartbeats.
[ArduPilot switch behavior](https://ardupilot.org/copter/docs/common-auxiliary-functions.html#mode-switches)

The program treats any observed transition into GUIDED as the enable request,
including a transition requested by a ground station. It does not distinguish
which device requested the mode change.

## Command timing and failure behavior

Camera processing runs separately from the control loop. Only the latest
observation is retained, and live setpoints are sent at `10 Hz` while the controller
is enabled. Capture timestamps, rather than processing time, determine freshness.

| Event | Program response |
| :--- | :--- |
| Latest frame has no valid target | Request zero forward velocity and zero yaw rate on the next control tick. |
| No new valid command for `0.5 s` | Expire the cached command and request zeros. |
| Target remains lost for `2 s` | Latch following off; keep requesting zeros while still enabled in GUIDED. Target reappearance alone does not resume following. |
| Flight-controller heartbeat absent for `3 s` | Suspend transmission and latch following off. If heartbeats resume in GUIDED after an engaged session, only zeros are allowed until a mode cycle. |
| Camera worker fails | Exit. Restart the program and complete the mode cycle before following again. |
| Ctrl+C or another shutdown | Attempt a final zero command only if an engaged session still has a fresh heartbeat and reports armed GUIDED; then close resources. |

Zero commands request a stop; they do not confirm that the aircraft is hovering.
ArduCopter also has its own configurable `GUID_TIMEOUT` for lost velocity commands
(default `3 s`). The program's heartbeat timeout is separate from this flight
controller setting. [ArduCopter Guided timeout](https://ardupilot.org/copter/docs/ac2_guidedmode.html)

Target selection starts with a person detection and associates subsequent
detections using proximity to the previous 3D position. This is not person
recognition: crossing people, occlusion, or reacquisition can select another person.
After 10 consecutive processed frames miss an acquired target, the visual tracker
stops acquiring until reset. This can happen before the control loop's `2 s`
timeout. In live mode, a new out-of-GUIDED/into-GUIDED cycle resets acquisition;
in dry-run mode, restart the program. Each new live session discards observations
captured before enabling. The coordinate conversion assumes a level camera facing
forward along the aircraft.

## Verification

Run the automated checks without connecting flight hardware:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

These checks exercise software behavior with simulated inputs. OAK-D Lite/Pi
hardware verification and ArduCopter SITL testing remain to be completed before
flight use.

## Parts
