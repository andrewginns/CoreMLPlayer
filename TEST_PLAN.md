# CoreMLPlayer Test Plan

This test plan focuses on preventing regressions from the planned Core ML/Vision pipeline changes (request reuse, error handling, orientation handling, async model loading, and pixel format tuning). It covers new unit and integration tests plus manual checks where automation is difficult.

## 1. Test Targets and Fixtures
- **Add a `CoreMLPlayerTests` target** with XCTest.
- **Fixtures**: small sample video (landscape) and portrait stills for orientation checks; synthetic pixel buffers in `kCVPixelFormatType_32BGRA` and model-preferred formats; a lightweight mock `MLModel` and `VNCoreMLModel` wrapper for failure injection.
- **Environment**: ensure tests run on device and simulator; gate GPU-dependent tests with `XCTSkip` when `MTLDevice` is unavailable.

## 2. Model Loading & Configuration
- **Async load success**: `MLModel.load(contentsOf:configuration:)` resolves and returns a non-nil model; configuration respects `computeUnits` and `allowLowPrecisionAccumulationOnGPU`.
- **Compilation fallback**: when passed an uncompiled model URL, verify the code compiles once and caches the compiled URL; subsequent loads reuse the compiled path.
- **Error surfacing**: inject an invalid model and assert the load path propagates errors (no silent failures).
- **Warm-up inference**: after loading, assert a single warm-up request executes without throwing and marks the model as ready for reuse. YOLOv3Tiny emits a benign console warning about missing `precisionRecallCurves` because it is not an updatable model; the test should tolerate this log as long as the request succeeds.

## 3. Vision Request Lifecycle
- **Request reuse**: repeated video frames should reuse the same `VNCoreMLRequest` / `VNSequenceRequestHandler`; track allocations or identifiers to ensure no per-frame recreation.
- **Error handling**: simulate a Vision error (e.g., mismatched pixel buffer attributes) and assert errors are logged/propagated, not swallowed.
- **Crop/scale compliance**: validate `imageCropAndScaleOption` matches the model’s expected input; tests check `.centerCrop` (or configured option) is applied rather than a hard-coded default.

## 4. Orientation & Pixel Buffer Handling
- **Orientation mapping**: feed landscape and portrait CMSampleBuffers and assert the resolved `CGImagePropertyOrientation` passed to Vision matches the video track orientation.
- **Pixel format selection**: when `idealFormat` is available, `AVPlayerItemVideoOutput` should be initialized with matching pixel buffer attributes; tests confirm the requested attributes and that fallback to BGRA occurs when unsupported.
- **Pixel buffer validity**: confirm the request handler rejects nil or stale pixel buffers and surfaces an error.

## 5. Inference Output Validation
- **Deterministic outputs**: run inference on a fixed fixture image and assert detections (labels/bounding boxes) stay within a tolerance envelope to catch pre/post-processing regressions.
- **Performance budget**: use `measure` blocks to ensure per-frame inference time stays within the real-time threshold (e.g., <33ms on a supported device) after changes.
- **Max-FPS loop stability**: stress test with rapid frame delivery to ensure no memory growth (via allocations audit) and no dropped frames due to request recreation.

## 6. UI/Integration Checks
- **Playback overlay**: snapshot tests for the detection overlay view to ensure bounding boxes render in correct orientation and aspect after crop/scale changes.
- **Lifecycle**: start/stop playback multiple times to verify requests/handlers are released and recreated cleanly; assert no crashes when the app re-enters foreground.
- **Error messaging**: inject model load failures and Vision errors and assert user-facing alerts or logs appear (and do not block main thread responsiveness).

## 7. Manual Verification
- **On-device smoke test**: run the app on a physical device, confirm stable FPS, correct bounding boxes, and no UI hangs when toggling models or playback speed.
- **GPU/CPU toggle**: switch compute units (CPU-only vs. ANE/GPU) to observe consistency and ensure the app handles hardware differences gracefully.

## 8. Automation Hooks
- Integrate tests into CI with separate schemes for unit/integration and UI snapshot runs.
- Capture performance baselines in CI artifacts for regression tracking.
- Provide toggles/env vars to skip GPU-reliant tests on unsupported runners.
