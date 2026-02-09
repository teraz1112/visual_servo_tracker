param(
  [string]$Config = "configs/offline_sample.yaml",
  [switch]$VerboseLog
)

$env:PYTHONPATH = "src"
$cmd = @("-m", "visual_servo_tracker.cli", "offline-run", "--config", $Config)
if ($VerboseLog) { $cmd = @("-m", "visual_servo_tracker.cli", "--verbose", "offline-run", "--config", $Config) }
python @cmd
