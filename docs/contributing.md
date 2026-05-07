# Contributing

This repository is now Swift-first.

Before sending changes, run:

```bash
swift build --disable-sandbox --jobs 18
SWIFT_BUILD_JOBS=18 scripts/verify_swift_port.sh
```

For changes that touch the real MLX bridge, also run the real backend smoke when the local model is available:

```bash
SWIFT_BUILD_JOBS=18 scripts/verify_real_gemma4_smoke.sh
```
