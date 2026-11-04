# Sensor record v1

Status: an acquisition specification and an offline Python reader. No Android
recorder or real participant recording is included in this module.

Each recording consists of a metadata JSON file and a UTF-8 CSV event file.
One pair describes exactly one participant, session, device, placement, and
stable activity. File paths are supplied explicitly to the offline reader.

## Metadata

Required fields are shown in this **synthetic** example. Values describing a
real device must be filled from that device, not copied from this example.

```json
{
  "schema_version": "sensor-record-v1",
  "session_id": "p01-walking-01",
  "participant_id": "p01",
  "purpose": "development",
  "activity_id": 1,
  "device_model": "synthetic example",
  "os_version": "synthetic example",
  "placement": "waist",
  "orientation": "portrait, top toward head, screen outward",
  "accelerometer_unit": "m/s^2",
  "gyroscope_unit": "rad/s",
  "requested_sampling_hz": 50,
  "stable_start_ns": 1000000000,
  "stable_end_ns": 61000000000
}
```

- `activity_id` uses the original UCI IDs 1–6 in
  [model-interface-v1.md](model-interface-v1.md).
- `purpose` is `development` or `final_test`. Assign it before collection and
  keep the whole session in that role. Development includes debugging and any
  adaptation. Final-test sessions remain unopened until the method is frozen.
- `stable_start_ns` and `stable_end_ns` use the same monotonic device clock as
  the events; both endpoints are inclusive and start must precede end. Confirm and trim the stable interval
  before evaluation. A stable activity change requires a new session.
- Optional fields: sensor manufacturer/name, app version, collection date,
  placement photograph reference without identifying information, and notes.
- IDs must be anonymous. Store participant consent and any identity mapping
  outside Git. Raw personal recordings also stay outside Git.

## Event CSV

Exact header:

```csv
timestamp_ns,sensor,x,y,z,accuracy
1000000000,accelerometer,0.0,0.0,9.80665,3
1001000000,gyroscope,0.0,0.0,0.0,3
```

`sensor` is `accelerometer` (`TYPE_ACCELEROMETER`, includes gravity) or
`gyroscope` (`TYPE_GYROSCOPE`). Axes retain native device order. Values must be
finite. `accuracy` preserves Android's integer accuracy status (-1 through 3).
Use sensor-event timestamps, not delivery times or wall-clock time. Timestamps
must increase strictly within each sensor stream; cross-sensor rows may be
interleaved or arrive in either order. Do not fabricate a paired reading when
only one sensor reports an event. Recording must retain the original readings
before interpolation, unit conversion, and standardization.

The reader rejects unknown sensors, duplicate or decreasing per-sensor
timestamps, invalid units or labels, malformed rows, and missing sensors.
Observed sampling rates and timestamp gaps are reported per sensor. They are
timing diagnostics, not an exact count of lost samples.

## Collection protocol

1. Use the same available phone and fixed waist position for the initial
   study. Record its actual model, system, orientation, and sensor information.
2. Start both sensors and inspect stationary readings and actual sample timing.
   Confirm the accelerometer retains gravity. Requested 50 Hz is not a measured
   guarantee; keep event timestamps.
3. Use the preassigned anonymous session ID and purpose in
   `ml/configs/phone-sessions-v1.csv`. Obtain the participant's agreement to
   record motion signals and stop if an activity cannot be performed safely.
4. Aim for 60 seconds of stable activity per session, excluding setup and
   transitions. Keep both sensors recording through the entire stable interval.
   Record stairs only where practical; do not substitute a different activity
   while retaining its label.
5. Stop, annotate the stable interval, and save metadata plus raw events. Check
   timing diagnostics, finite values, duration, and class/placement metadata.
6. Keep planned, collected, excluded, and evaluated states distinct in the
   session registry. Log the reason for any exclusion; do not replace a session
   because of its classification score.

The initial registry plans two anonymous participants, six activities, and
three separate sessions per activity: two development sessions and one final
test session. All 36 entries start as `planned`; none implies data was collected.
This is a small, session-separated phone check. Because the same participants
can appear in both roles, it is not a subject-independent phone study.
