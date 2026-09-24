# Official MVTec AU-PRO evaluation

This is an inference/evaluation pipeline, not a training entry point. It does
not edit the paper or use our historical strict evaluator for metrics.

## Official code

`third_party/mvtec/SOURCE.json` records the URLs/hashes of the two archives
downloaded directly from MVTec's AD and AD2 pages. The extracted code is
unchanged. The AD2 README recommends the AD evaluator for public-test AU-PRO.
`pro_curve_util.compute_pro` and `generic_util.trapezoid` are called directly;
the existing integration-limit parameter handles 0.05 and 0.30.

## Public evaluation

After training, place each selected checkpoint under the
`saved_results/INP-Former-Multi-Class-8cat-sweep_...INP_num=M_.../model.pth`
name written by the trainer. The helper requires those checkpoints and
prints the export and evaluation jobs before submission:

```bash
python scripts/submit_ad2_official_eval.py \
  --geometry full-frame --run-tag YOUR_RUN_TAG --models 6 2 1
```

Add `--submit` to enqueue the printed jobs. The helper rejects an existing
output directory or job ledger; choose a new run tag for a new evaluation.
`--after-job JOB_ID` can make all exports wait for a successful validation job.
Each model has L4 inference followed by a CPU-only 128-GiB evaluation job.
Both cutoffs share identical predictions. Results are under
`evaluation_results/YOUR_RUN_TAG/full-frame/M<M>/metrics/summary.json`
(scale 0..1). Exports require the companion log to contain an epoch-200
evaluation on all categories.

Inference exports every public good/bad image, float16 native-resolution TIFFs,
float32 model256 maps, and original top-1%-pixel image scores. Manifests record
checkpoint, image, mask and TIFF hashes, geometry, dtype and source code.
The evaluator reads the exact exported TIFFs, includes normal images, evaluates
each category independently, then averages categories equally. I-AUROC uses
the recorded top-1% model score; max-TIFF-score I-AUROC is separately reported.
P-AUROC uses native pixel scores and ground truth. No hidden-test local metrics.

### Explicit geometry choice

- `full-frame`: resize the complete image to 392x392 at inference. Restore its
  smoothed256x256 map to native H/W. This is a **changed inference protocol**
  compared to the historical 448->392 center crop. It preserves full-image
  coverage, uses no invented border scores and is suitable for public/private
  map export. Treat resulting scores as new results, not evaluator-only changes.
- `legacy-crop256`: reproduce the historical input/mask geometry, while using
  official AU-PRO and retaining normal images. This isolates evaluator changes
  on the observed crop. It is diagnostic only, not native full-image evaluation.
  It cannot be used to generate private/server submissions.

Neither mode alters official AU-PRO algorithms. Their results must not be mixed.

## Historical recovery

The new AD2 models cannot recover historical seven-category results;
the new MVTec loss-ablation models cannot replace the historical MVTec baseline.
Once an exact historical full-model checkpoint is recovered, the exporter
supports `--dataset AD --split test` and `--dataset VisA --split test`, as well
as AD2 public input. For historical AD2 pass an explicit seven-category list.
Do not use `--require-epoch200` unless the accompanying historical log matches
that convention. Strict state loading remains mandatory.

Continuous maps from an external run must cover every normal/anomalous image,
have correct IDs and known geometry, and include preprocessing provenance. Anomalous-only
caches, binary masks, plots and scalar summary metrics are insufficient.

## Later private export and preparation

After choosing and freezing a final model, run `scripts/export_ad2_official.sh`
with arguments `M CHECKPOINT NEW_OUTPUT full-frame test_private`, and separately
with `test_private_mixed`. This is inference only. Use the same input protocol,
weights and map processing as public evaluation. Provide both private split datasets locally for all eight categories.

Stage a clean submission directory (no upload) from those two manifests:

```bash
python scripts/prepare_ad2_server_submission.py \
  --manifests PRIVATE_OUTPUT/manifest.json MIXED_OUTPUT/manifest.json \
  --output NEW_SERVER_DIRECTORY
python third_party/mvtec/MVTecAD2_public_code_utils/check_and_prepare_data_for_upload.py \
  NEW_SERVER_DIRECTORY
```

Run these inside the existing container. The first command hard-links identical
continuous float16 TIFFs; the second is MVTec's original checker/packager.
Thresholded masks are optional for AU-PRO. Server submission itself is not
performed by any of these scripts.

## Verification

`python scripts/test_mvtec_official_adapter.py` checks exact equality with the
official file-based evaluator, tied-score/cutoff behavior, good-image inclusion,
complete manifests, finite maps and known perfect-prediction results.
`scripts/smoke_official_export.sh` additionally loads a real complete AD2 model,
exports two images in both geometries on CPU, and runs the metric tests in the
actual training container. Smoke manifests are intentionally incomplete and
cannot be accepted as paper evaluations.
