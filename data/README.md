# Data layout

- `data/samples/`: tracked minimal sample inputs used by smoke tests.
- `outputs/`: generated runtime outputs (`jacobian`, `graph`, `optimized`, `runs`), excluded by `.gitignore`.

To add your own dataset:
1. Create `data/samples/<version>/goal/0_0.jpg`.
2. Create `data/samples/<version>/gap/<dx>_<dy>.jpg` files.
