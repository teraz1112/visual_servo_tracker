# Third-party policy

External repositories (e.g. DINOv3/SAM) should be placed under `third_party/` only when required.

Rules:
1. Remove nested `.git` metadata before publishing this project.
2. Keep upstream `LICENSE` files in each imported package.
3. Update `THIRD_PARTY_LICENSES.md` and `NOTICE`.

Use helper script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/cleanup_third_party_git.ps1
```
