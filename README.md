# README: Monocular Depth Estimation
## 1. Overview 🧠
This directory implements **monocular depth estimation** for Task 3, which predicts dense depth maps from single RGB images using deep learning. Built on PyTorch and evaluated on ScanNet, it supports training, inference, quantitative evaluation, visualization, and fair comparison between a baseline model and modern foundation models (DA3 / VGGT).

## 2. Technical Background 📚
Monocular depth estimation is an ill-posed computer vision task: a single 2D image does not provide explicit 3D geometry. The key ideas are:
- **Feature Extraction** 🧬: A ResNet50 backbone extracts multi-level visual features including edges, textures, and structural patterns.
- **Depth Regression** 📏: A decoder head maps features to a dense depth map with the same resolution as the input.
- **Scale-Shift Alignment** ⚖️: Resolves scale ambiguity by aligning predictions with ground-truth depth for consistent evaluation.
- **Loss & Metric** 📊: Trained with depth loss; evaluated by **AbsRel** (lower = better).

## 3. File Structure 📁
| File Name | Description |
|----------|-------------|
| `depth_model.py` | Defines the baseline depth model (ResNet50 + decoder) |
| `scannet_dataset.py` | ScanNet data loader, preprocessing, and masking |
| `metrics.py` | AbsRel metric and scale-shift alignment functions |
| `train.py` | Training pipeline with loss, optimizer, and checkpoint saving |
| `infer.py` | Inference for single-image depth prediction |
| `visualize.py` | Visualization and comparison for Baseline / VGGT / DA3 |

## 4. Usage 🚀
### 4.1 Environment Setup
```bash
pip install torch torchvision numpy matplotlib opencv-python scipy
```

### 4.2 Model Training 🏋️‍♂️
```bash
python train.py \
  --scannet_root /path/to/scannet \
  --epochs 32 \
  --batch_size 16 \
  --image_size 240 320
```

### 4.3 Inference & Visualization 🎨
```bash
python visualize.py \
  --scannet_root /path/to/scannet \
  --checkpoint ./trained_models/best.pth \
  --scene_name scene0001_01
```

## 5. Evaluation Metric 📐
The main metric is **Absolute Relative Error (AbsRel)**:
$$\text{AbsRel} = \frac{1}{N}\sum\frac{|d_{pred} - d_{gt}|}{d_{gt}}$$
Lower values indicate higher depth accuracy.

## 6. Notes 💡
- Adjust `batch_size` if you encounter GPU out-of-memory issues.
- Set `min_depth` and `max_depth` appropriately for indoor scenes.
- Foundation models (DA3/VGGT) require `foundation_models.py`.
