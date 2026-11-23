# Within-recording LWIR segmentation in glioblastoma surgery

Code accompanying the brief report: *Long-Wavelength Infrared Image Segmentation in Glioblastoma Surgery* (manuscript in preparation).

A lightweight U-Net was trained separately for each surgical recording to segment expert-defined tumor regions in long-wave infrared (LWIR) frames. Frames from each recording were randomly split into training, validation and internal-test sets. Results therefore reflect within-recording agreement, not performance on unseen patients.

## Files

| File | Purpose |
|---|---|
| `extract_frames.py` | Extracts frames from multi-frame TIFF recordings and min-max normalizes them to 8-bit PNG images |
| `train_evaluate_unet.py` | Splits frames, trains the U-Net, evaluates on internal-test frames (Dice, ASSD) and saves overlays |
| `requirements.txt` | Python packages required |

## Requirements

```
pip install -r requirements.txt
```

## Usage

The script is run once per surgical recording.

1. **Extract frames:** in `extract_frames.py`, set the path to the TIFF recording and the output folder, then run it.
2. **Set paths:** in `train_evaluate_unet.py`, set `FRAMES_DIR` (extracted frames) and `MASKS_DIR` (binary reference masks, PNG). Frame and mask filenames must share the same frame number.
3. **Run:** `python train_evaluate_unet.py`

The script:
- splits frames 70/20/10 into training, validation and internal-test sets (random seed 42)
- trains the model with early stopping on validation Dice
- saves the best model (`best_fold0.pth`), the training log and curve, test-set Dice and ASSD, and overlay images

## Data availability

The intraoperative LWIR recordings and reference masks are stored on a secure University of Cambridge server and are not included in this repository. Data access requests may be directed to the corresponding author of the paper and are subject to institutional and ethical approval.

## License

MIT
