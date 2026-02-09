param(
  [Parameter(Mandatory=$true)][string]$Video,
  [Parameter(Mandatory=$true)][string]$Target,
  [Parameter(Mandatory=$true)][string]$Jacobian,
  [int]$IterationsPerFrame = 10
)

$env:PYTHONPATH = "src"
python -m visual_servo_tracker.cli track-video --video $Video --target $Target --jacobian $Jacobian --iterations-per-frame $IterationsPerFrame
