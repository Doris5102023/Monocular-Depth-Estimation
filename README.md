# README: Monocular Depth Estimation
## 1. Overview 🧠
This repository implements **monocular depth estimation** , which predicts dense depth maps from single RGB images using deep learning. Built on PyTorch and evaluated on the [ScanNet dataset](https://github.com/ScanNet/ScanNet), it supports:
- Training a custom ResNet50 baseline model
- Inference & quantitative evaluation (AbsRel metric)
- Qualitative visualization of depth predictions
- Performance comparison with VGGT/DA3 foundation models

## 2. Technical Background 📚
Monocular depth estimation is an ill-posed computer vision task: a single 2D image does not provide explicit 3D geometry. The key ideas are:
- **Feature Extraction** 🧬: A ResNet50 backbone extracts multi-level visual features including edges, textures, and structural patterns.
- **Depth Regression** 📏: A decoder head maps features to a dense depth map with the same resolution as the input.
- **Scale-Shift Alignment** ⚖️: Resolves scale ambiguity by aligning predictions with ground-truth depth for consistent evaluation.
- **Loss & Metric** 📊: Trained with depth loss; evaluated by **AbsRel** (lower = better).

## 3. File Structure 📁
| File Name               | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `depth_model.py`        | ResNet50 baseline model definition (backbone + depth regression head).      |
| `scannet_dataset.py`    | ScanNet data loader (preprocessing + valid depth mask generation).          |
| `metrics.py`            | AbsRel metric + scale-shift alignment functions.                           |
| `train.py`              | Training pipeline (loss computation / optimizer / checkpoint saving).       |
| `test.py`               | Test-time evaluation (AbsRel calculation on test set).                     |
| `infer.py`              | Single-image depth prediction.                                              |
| `visualize.py`          | Baseline model visualization (RGB → GT → Pred → Error).                    |
| `visualize_compare.py`  | Side-by-side comparison (Baseline/VGGT/DA3).                                |
| `foundation_models.py`  | Wrapper for VGGT/DA3 inference. 

## 4. Foundation Model Resources 🤖
### VGGT
- Paper: [Vision Grounded Geometry Transformer](https://arxiv.org/abs/2503.11651)
- Model Zoo: [https://github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt)

### Depth Anything 3 (DA3)
- Paper: [Depth Anything 3: Recovering the Visual Space from Any Views](https://arxiv.org/abs/2511.10647)
- Model Zoo: [https://github.com/ByteDance-Seed/depth-anything-3](https://github.com/ByteDance-Seed/depth-anything-3)

## 5. Usage 🚀
### 5.1 Environment Setup
```bash
pip install torch torchvision numpy matplotlib opencv-python scipy
```

### 5.2 Model Training 🏋️‍♂️
```bash
python train.py \
  --scannet_root /path/to/scannet \
  --output_dir /path/to/save \
  --train_split_file /path/to/scannetv2_train.txt
```

### 5.3 Model Evalusation 🎀
Baseline:
```bash
python test.py \
  --scannet_root /path/to/scannet \
  --save_json /path/to/save \
  --split_file /path/to/scannetv2_val.txt \
  --checkpoint ./trained_models/best.pth 
```
VGGT:
```bash
python compare.py \
  --scannet_root /path/to/scannet \
  --save_json /path/to/save \
  --split_file /path/to/scannetv2_val.txt \
  --model vggt
```
DA3:
```bash
python compare.py \
  --scannet_root /path/to/scannet \
  --save_json /path/to/save \
  --split_file /path/to/scannetv2_val.txt \
  --model da3 \
  --da3_model_type DA3NESTED-GIANT-LARGE
```
[Beseline vusualizetion result](https://github.com/Doris5102023/Monocular-Depth-Estimation/blob/main/depth_qualitative_visualization_1.png)
[Comparison vusualizetion result](https://github.com/Doris5102023/Monocular-Depth-Estimation/blob/main/depth_qualitative_visualization_1.png)

### 5.4 Inference & Visualization 🎨
```bash
python visualize.py \
  --scannet_root /path/to/scannet \
  --checkpoint ./trained_models/best.pth \
  --scene_name scene0001_01 \
  --save_path depth_qualitative_visualization_1.png
```

## 6. Evaluation Metric 📐
The main metric is **Absolute Relative Error (AbsRel)**:
$$\text{AbsRel} = \frac{1}{N}\sum\frac{|d_{pred} - d_{gt}|}{d_{gt}}$$
Lower values indicate higher depth accuracy.

## 7. Notes 💡
- Adjust `batch_size` if you encounter GPU out-of-memory issues.
- Set `min_depth` and `max_depth` appropriately for indoor scenes.
- Foundation models (DA3/VGGT) require `foundation_models.py`.
