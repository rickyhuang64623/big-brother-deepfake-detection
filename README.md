# Big Brother is Watching: Proactive Deepfake Detection via Learnable Hidden Face

This repository contains the official implementation of  
**"Big Brother is Watching: Proactive Deepfake Detection via Learnable Hidden Face"**  
by Hongbo Li, Shangchao Yang, Ruiyang Xia, Lin Yuan, Xinbo Gao (2024).

## Overview

This project introduces a proactive deepfake detection framework that embeds a learnable face template into images using an invertible neural network (INN) for steganography. The system enables robust detection of manipulated images by extracting and analyzing the hidden template, even after common image manipulations or forgeries.

**Key components:**
- **Invertible Steganography Network:** Hides and extracts a learnable face template within images.
- **Template Learning Module:** Optimizes the hidden face template to resemble a realistic average face.
- **Detection Network:** Identifies whether an image is authentic or manipulated by analyzing the extracted template.

## Features

- End-to-end training of embedding, template, and detection networks.
- Patch-based detection for high-resolution images.
- Visualization tools for template evolution and detection results.
- Support for benign and malicious manipulations (JPEG, blur, self-blending, etc.).
- Logging and experiment tracking with [Weights & Biases](https://wandb.ai/).

## Project Structure

```
.
├── data/                   # Datasets (train/val/test)
├── models/                 # Model definitions (steganography, template, detector)
├── train/                  # Training scripts
├── test/                   # Testing and evaluation scripts
├── scripts/                # Dataset preparation, average face computation
├── outputs/                # Saved templates, visualizations, results
├── checkpoints/            # Model checkpoints
├── configs/                # YAML configuration files
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rickyhuang64623/big-brother-deepfake-detection.git
   cd big-brother-deepfake-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset:**
   - Place your source images in `img_align_celeba/`.
   - Run:
     ```bash
     python test.py
     ```
   - This will split your images into `data/train/` and `data/test/` folders.

4. **Compute the average face (for template initialization):**
   ```bash
   python scripts/compute_average_face.py
   ```

## Training

Train the full pipeline (steganography, template, detector) with:
```bash
python train/train.py --config configs/train_config.yaml
```
- Checkpoints and template images will be saved in `checkpoints/` and `outputs/`.

## Inference & Detection

**Detect hidden templates and manipulations in an image:**
```bash
python detect.py --input path/to/image.jpg --checkpoint checkpoints/checkpoint_epoch_100.pth --config configs/train_config.yaml
```
- Outputs a detection report and visualization highlighting suspicious regions.

**End-to-end test (embedding, extraction, detection):**
```bash
python test/protect_and_detect.py --config configs/train_config.yaml --checkpoint checkpoints/checkpoint_epoch_100.pth --mode protect --input path/to/image.jpg --output outputs/protected.jpg
python test/protect_and_detect.py --config configs/train_config.yaml --checkpoint checkpoints/checkpoint_epoch_100.pth --mode detect --input outputs/protected.jpg --output outputs/
```

## Visualization

- **Template evolution:**  
  Visualize the learned template at each epoch (see `outputs/template_epoch_*.png`).
- **Hidden face demonstration:**  
  ```bash
  python show_hidden_face.py
  ```
- **Template visualization:**  
  ```bash
  python visualize_template.py
  ```

## Configuration

All hyperparameters and paths are managed via YAML files in `configs/`.  
Modify `configs/train_config.yaml` to adjust training settings, loss weights, and model parameters.

## Pre-trained Models

Pre-trained checkpoints can be downloaded (if available) or trained from scratch.  
To use your own models, place them in the `checkpoints/` directory.

## Citation

If you use this codebase or ideas from our paper, please cite:
```bibtex
@article{li2024bigbrother,
  title={Big Brother is Watching: Proactive Deepfake Detection via Learnable Hidden Face},
  author={Li, Hongbo and Yang, Shangchao and Xia, Ruiyang and Yuan, Lin and Gao, Xinbo},
  journal={arXiv preprint arXiv:2504.11309},
  year={2024}
}
```

## License

This repository is released under the MIT License.

## Contact

For questions or collaborations, please open an issue or contact the authors. 