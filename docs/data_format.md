# Data format

Required layout per version:

```text
data/samples/<version>/
├─ goal/0_0.jpg
└─ gap/<dx>_<dy>.jpg
```

`<dx>_<dy>.jpg` must represent shifted crops relative to `goal/0_0.jpg`.
