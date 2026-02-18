param(
  [string]$Config = "configs/offline_sample.yaml",
  [int]$IterationsPerFrame = 10,
  [switch]$VerboseLog
)

$env:PYTHONPATH = "src"
if ($VerboseLog) {
  python -m visual_servo_tracker.cli --verbose offline-track-video --config $Config --iterations-per-frame $IterationsPerFrame
} else {
  python -m visual_servo_tracker.cli offline-track-video --config $Config --iterations-per-frame $IterationsPerFrame
}
