# CCTV Anomaly Classification

A Streamlit application that classifies one sampled segment of an uploaded MP4 using a ViViT video classifier, accompanied by a fine-tuning and evaluation notebook. ([app.py](app.py), `main`; [notebook](cctv-anomaly-classification.ipynb), cells 7–19.)

Notebook cell references below are **one-based positions, counting both code and Markdown cells**, not execution counts.

## Features

- Upload and preview an MP4, decode selected frames with PyAV, and run a neural-network forward pass without gradient computation. ([app.py](app.py), `main`, `read_video_pyav`.)
- Select the highest-scoring model label and display a button for one of seven supported labels: `abuse`, `arson`, `burglary`, `explosion`, `normal`, `roadaccidents`, and `shooting`. The label mapping comes from the loaded model configuration; the UI handles these seven strings explicitly. ([app.py](app.py), `main`; [notebook](cctv-anomaly-classification.ipynb), cell 5.)
- Display a predefined description when the label button is clicked. These descriptions are static text, not generated explanations of the uploaded footage. ([app.py](app.py), `main`, `stream_text`.)
- Fine-tune a pretrained ViViT classifier, compute classification metrics, plot a test confusion matrix, and explicitly upload the trained model to Hugging Face from the notebook. ([notebook](cctv-anomaly-classification.ipynb), cells 7, 13–22.)

## How it works

The application follows this path in [app.py](app.py):

1. `main` opens the upload and reads the first video stream's reported frame count.
2. `sample_frame_indices` selects 32 indices from a randomly positioned 128-frame span using `linspace` and clipping. This is one segment, not a scan of the whole recording.
3. `read_video_pyav` seeks to the beginning, decodes through the selected segment, and stacks the selected frames as RGB arrays.
4. `load_model`, decorated with `st.cache_data`, loads the image processor and `VivitForVideoClassification` from `ratchy-oak/vivit-b-16x2-kinetics400-finetuned-cctv-surveillance`.
5. `main` processes the frames, calls the model under `torch.no_grad()`, takes `argmax` over its logits, and looks up `model.config.id2label`.

This is a trained neural-network classification workflow. The notebook initializes from `google/vivit-b-16x2-kinetics400`, supplies a seven-class mapping, and calls `Trainer.train()`. It uses `Trainer.predict(test_dataset)` for the confusion matrix and `Trainer.evaluate()` for validation metrics. ([notebook](cctv-anomaly-classification.ipynb), cells 7, 15, 17, 19.)

## Tech stack

| Component | Libraries actually used | Evidence |
| --- | --- | --- |
| Upload interface and playback | Streamlit | [app.py](app.py), `main` |
| Video decoding and sampling | PyAV, NumPy | [app.py](app.py), `read_video_pyav`, `sample_frame_indices` |
| Model loading and inference | PyTorch, Transformers | [app.py](app.py), `load_model`, `main` |
| Training data and transforms | pandas, scikit-learn, PyTorchVideo, torchvision | [notebook](cctv-anomaly-classification.ipynb), cells 3–11 |
| Training and evaluation | Transformers Trainer, Evaluate, scikit-learn, Matplotlib | [notebook](cctv-anomaly-classification.ipynb), cells 13–19 |
| Notebook preview and publishing | imageio, IPython, huggingface_hub | [notebook](cctv-anomaly-classification.ipynb), cells 11, 21–22 |

The application dependencies in [requirements.txt](requirements.txt) have **no version pins**. The notebook records Python 3.10.13 in its metadata, but this is historical environment metadata, not a declared supported-version range or a reproducible dependency lock.

## Setup and usage

From the repository directory, create a Python environment, install the application requirements, and launch its Streamlit entry point:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

These commands correspond to [requirements.txt](requirements.txt) and [app.py](app.py), which calls `main` under its `__main__` guard. They are a source-grounded launch procedure; a clean installation with newly resolved dependency versions has not been validated. The model and processor must be available from Hugging Face or its local cache, and their revision is not pinned by `load_model`. ([app.py](app.py), `load_model`.)

In the interface, upload an MP4, wait for classification, and click the resulting label button to read its static description. [video.mp4](video.mp4) is a bundled input you can try; the application does not load it automatically. Use a decodable file whose first video stream reports **more than 128 frames**—the sampler fails otherwise. ([app.py](app.py), `main`, `sample_frame_indices`.)

### Training notebook

Open [cctv-anomaly-classification.ipynb](cctv-anomaly-classification.ipynb) in a notebook environment with the external dataset mounted at:

```text
/kaggle/input/real-time-anomaly-detection-in-cctv-surveillance/
```

Cell 3 expects `data/train.csv` and `data/test.csv`, each containing `label` and `video_name`, plus the referenced videos. Cell 5's `correct_file_path` hardcodes the dataset root even though it accepts a `root_path` parameter; adapting the notebook requires updating this function as well as cell 3. The dataset is not included in the [repository files](.).

Cell 1 installs only `pytorchvideo`, `transformers`, and `evaluate`. The later imports also require the notebook libraries listed above and `wandb`; [requirements.txt](requirements.txt) is only the application's dependency list. No complete, pinned training environment is supplied. Cells 21–22 perform interactive Hugging Face login and `trainer.push_to_hub()`; skip those cells if you only want local training and evaluation. ([notebook](cctv-anomaly-classification.ipynb), cells 1, 3–22.)

## Data and evaluation limits

The notebook reads an external dataset rather than collecting footage itself. Its saved cell 3 output lists 14 labels. Cell 5 keeps seven labels, combines the original train and test tables, discards a stratified portion, and creates new row-level splits with `random_state=42`. ([notebook](cctv-anomaly-classification.ipynb), cells 3–5.)

The saved outputs imply 1,400 selected metadata rows before subsampling: 950 `normal`, 150 `roadaccidents`, 100 `burglary`, and 50 each for the other four selected classes. Cell 5 reports **1,008 training, 226 test, and 26 validation rows**. These are historical metadata-row counts, not independently verified counts of unique recordings, cameras, locations, incidents, or people. The CSVs and video corpus are absent, so source independence and duplicate leakage cannot be checked. The split code does not group by physical source. ([notebook](cctv-anomaly-classification.ipynb), cells 3–5; [repository files](.).)

Validation and test datasets use uniform clip sampling. Metrics operate on the resulting clip predictions without a per-video aggregation step, so metadata rows and evaluated clips are different units. `trainer.evaluate()` uses the validation dataset already used for model selection, not `test_dataset`. The notebook contains saved results and the code intended to calculate them, but this checkout alone does not provide the data, split manifests, environment lock, or trained artifacts needed to reproduce them. No benchmark score is asserted here. ([notebook](cctv-anomaly-classification.ipynb), cells 11, 13, 15, 17, 19; [repository files](.).)

## Known limitations

- **Single-segment classification:** no live-camera input, temporal event localization, whole-video aggregation, alert delivery, confidence display, or unknown-class rejection is implemented in the upload flow. It always selects an `argmax` label; unsupported label strings have no UI fallback. ([app.py](app.py), `main`.)
- **Fragile input handling:** short videos and unknown/zero reported frame counts cause the sampler to fail. There is no application-level handling for decoder errors, missing video streams, failed model loading, or incomplete frame extraction; the opened container is not explicitly closed. ([app.py](app.py), `sample_frame_indices`, `read_video_pyav`, `main`.)
- **Different training and inference preprocessing:** training uses temporal subsampling, division by 255, normalization, scaling/resizing and flipping; validation uses resizing without the training augmentations. The application instead delegates image preprocessing to the downloaded processor and uses a different temporal span. Their equivalence is not enforced. ([notebook](cctv-anomaly-classification.ipynb), cell 9; [app.py](app.py), `main`.)
- **Limited reproducibility:** dependency versions and model revisions are unpinned, and dataset source identities are unavailable. Saved notebook outputs are historical evidence, not a fresh benchmark or a guarantee of deployment accuracy. ([requirements.txt](requirements.txt); [app.py](app.py), `load_model`; [notebook](cctv-anomaly-classification.ipynb), cells 3–5, 15–19.)
- **Verification scope:** the source path from sampling through model logits was traced, and an offline smoke check with the locally cached checkpoint successfully processed [video.mp4](video.mp4). This verifies execution, not prediction correctness. Browser interaction, clean installation, and retraining were not verified. No automated test files or test framework configuration are supplied in the [repository files](.); the notebook's “Model Testing” section is model evaluation code, not a software test suite. ([app.py](app.py), `main`; [notebook](cctv-anomaly-classification.ipynb), cell 17.)
