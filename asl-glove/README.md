# ASL Gesture Recognition & Voice Translation Glove

A wearable glove that recognizes static American Sign Language (ASL) alphabet
gestures and speaks them aloud. Built around an ESP32, DIY flex sensors, and
an on-device Random Forest classifier.

## Hardware

- **ESP32** dev board
- 4x DIY foam-based flex sensors (+ plans for a 5th thumb sensor and a
  copper-tape touch switch to disambiguate U/V)
- **MPU-6050** IMU for orientation
- **DFPlayer Mini** → amplifier → speaker for audio output
- Hybrid sensing: foam pressure pads + half-cut flex sensors + IMU

## Pipeline

1. **Data collection** — `python/collect_data.py` logs sensor readings per
   letter (static pose) over serial into `data/<date>/session_*.csv`.
2. **Cleaning / merging** — `python/clean_dataset_strict.py`,
   `python/repair_index_flex.py`, and `python/merge_data.py` combine and
   sanitize sessions into `data/combined_dataset.csv` and
   `data/ASL_Gesture_Dataset_Cleaned.csv`.
3. **Analysis** — `python/plot.py` produces the dataset EDA and train/test
   split under `dataset_analysis/`.
4. **Training** — `python/train_random_forest.py` trains the Random Forest
   classifier (the model actually deployed on-device) and writes
   `models_rf/random_forest.joblib`, `models_rf/labels.txt`, and a
   C header (`rf_model.h`) for on-device inference. `python/train_model.py`
   (a TensorFlow neural-net alternative) was also explored during
   development — see `report_plots/` for the comparison.
5. **Calibration** — `python/calibrate.py` records per-sensor calibration
   snapshots (`calibration/`); `python/generate_calibration_header.py`
   converts the latest one into `calibration.h` for the firmware.
6. **Deployment** — `esp32/rf_model_deploy/model_deploy_rf/` is the firmware
   sketch that loads the RF model and calibration and speaks the recognized
   letter through the DFPlayer Mini.

## Results

Final Random Forest model, evaluated on a held-out recording session
(session-based split, avoiding same-session leakage) — see
`report_plots/metrics_summary.json`:

| Metric | Value |
|---|---|
| Accuracy | 93.1% |
| Macro F1 | 93.3% |
| Weighted F1 | 93.1% |

Full performance breakdown, confusion matrix, and feature importance are in
`report_plots/`.

## Repo layout

```
python/          data collection, cleaning, training, calibration scripts
esp32/           firmware sketches (data collection, RF deploy, NN deploy, touch test)
data/            raw sessions + merged/cleaned datasets
calibration/     per-session sensor calibration snapshots
models_rf/       trained Random Forest model + generated C header (deployed model)
dataset_analysis/ EDA plots and train/test split
report_plots/    final evaluation plots and metrics
```

## Notes

- `esp32/model deploy/` contains the neural-network deployment sketch kept
  for reference from an earlier experiment; the Random Forest model
  (`esp32/rf_model_deploy/`) is what's actually used.
- Gestures are collected as static poses, including J and Z, which are
  normally dynamic/traced letters in ASL and are flagged as more
  error-prone.
