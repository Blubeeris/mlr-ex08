# Leaf Classifier — live camera demo (MPC-MLR exercise 8)

**Open it here:** <https://blubeeris.github.io/mlr-ex08/>

Point your **phone's rear camera** at a leaf and the page classifies it live, entirely
in the browser — no server, no uploads. Works in Safari on iPhone (needs iOS 16.4 or
newer, because the WebAssembly build of ONNX Runtime needs SIMD).

You can also tap **Upload photo** to classify a picture from the camera roll or a file —
handy when the camera is not available or the connection is too slow for the model.

There is a QR code at the bottom of the page that encodes the address above, so you can
open the app on your phone straight from a desktop screen.

## What the page does

Everything comes from [`model_info.json`](model_info.json) — the same four steps the
training code used:

1. **Crop the central square** of the camera frame (the area between the corner marks).
2. **Resize** it to `input_size` × `input_size` = **96 × 96**.
3. **Scale** the RGB pixels to **0…1** (`/255`).
4. **Standardize** each channel: `(x - mean) / std` with
   `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`.

The tensor is laid out **CHW** and passed as `1 × 3 × 96 × 96` float32 to the input
`image` of [`leaf_model.onnx`](leaf_model.onnx). The model returns `logits`
(`1 × 4`), a softmax turns them into probabilities, and the probability of **every**
class — `apple`, `cherry`, `chestnut`, `maple` — is shown as a live bar chart
(about 5 inferences per second on a recent phone).

| | |
|---|---|
| Model | `leaf_model.onnx` — 4-class CNN exported from PyTorch 2.14 (ONNX IR v10, input `image`, output `logits`) |
| Runtime | [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) 1.20.1 (WASM, single thread) from jsDelivr, with unpkg as a fallback |
| Build | one `index.html`, plain JavaScript, no framework and no build step |

## Self-check

The **Run self-check** button on the page runs the browser's own pipeline over
[`check_image.jpg`](check_image.jpg) and compares the result with the reference
probabilities in [`check_expected.json`](check_expected.json). It reports
`PASS` when the largest difference for any class is below `1e-3`.

A NumPy/`onnxruntime` reference implementation reproduces `check_expected.json` to
~7·10⁻⁹, and the browser pipeline agrees with it to the same order (worst class
6.7·10⁻⁹ in headless Chromium) — so the preprocessing in `index.html` matches the
training-time preprocessing.

## Files

| File | Purpose |
|---|---|
| `index.html` | the whole app (camera, upload, preprocessing, inference, results, QR code) |
| `leaf_model.onnx` | the trained classifier |
| `model_info.json` | class names, input size, channel mean/std — read at runtime |
| `check_image.jpg`, `check_expected.json` | self-check fixture and its reference output |
| `README.md` | this file |

## Notes on the implementation

* **Camera.** `getUserMedia({ video: { facingMode: { ideal: "environment" } } })`, and
  the `<video>` is `playsinline muted` so iOS plays it inline. Camera access needs a
  secure context, which GitHub Pages provides over HTTPS.
* **Safari.** WASM SIMD is feature-detected and the page says so if the browser is too
  old. `ort.env.wasm.numThreads = 1` keeps the runtime single-threaded, so it does not
  need `SharedArrayBuffer` — GitHub Pages does not send the COOP/COEP headers that
  `SharedArrayBuffer` would require (verified: the page loads and classifies with
  `crossOriginIsolated === false`).
* **Runtime version.** `leaf_model.onnx` has ONNX IR version 10 (PyTorch 2.14 export),
  which older onnxruntime builds refuse — 1.17 fails to parse it, so the page pins
  1.20.1 (the smallest WASM bundle that accepts the model; 1.22's bundle is twice the
  size). The first visit downloads ~17 MB (6 MB model + 11 MB runtime), then the browser
  caches it.
* **The QR code is generated in the page**, by a small byte-mode encoder written in
  `index.html` itself (byte mode, error-correction level M, versions 1–10, with Reed–
  Solomon error correction over GF(2⁵⁶)). No third-party QR image service is involved —
  the URL never leaves your browser.
* **Nothing is uploaded.** The frame is drawn to an offscreen 96 × 96 canvas, read back
  with `getImageData`, and fed to the local WASM model.

## Run it locally

GitHub Pages serves the branch over HTTPS, which is what the camera needs. Locally, any
static server works (the camera needs `localhost` or a trusted HTTPS origin):

```sh
python -m http.server 8000
# then open http://localhost:8000/
```

---

MPC-MLR, exercise 8 — leaf classification with ONNX Runtime Web.
