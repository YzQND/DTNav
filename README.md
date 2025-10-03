# Drone Visual Language Navigation System

A 3D drone navigation system that combines vision-language models (LLaVA-OneVision) with topological mapping for indoor navigation tasks. The system supports data collection, training, and real-time navigation with natural language instructions.

## Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Citation](#citation)

## Features

- **3D Drone Navigation**: Extended action space including vertical movements (move_up, move_down)
- **Visual-Language Grounding**: Uses LLaVA-OneVision for instruction following
- **Topological Mapping**: SigLIP-aligned feature extraction with FastSAM segmentation
- **Scene-Aware Navigation**: Multi-modal landmark recognition with scene understanding
- **Dual-Target Strategy**: Combines visible guide landmarks with final target landmarks
- **R2R Format Support**: Compatible with Room-to-Room dataset format

## System Requirements

- **GPU**: NVIDIA GPU with at least 24GB VRAM
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ for models and datasets
- **OS**: Linux (Ubuntu 22.04 recommended)
- **CUDA**: 11.7 or higher

## Installation

### 1. Environment Setup

```bash
# Create conda environment
conda create -n drone_vln python=3.9
conda activate drone_vln

# Install PyTorch (adjust for your CUDA version)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install Habitat-Sim (for data collection)

```bash
# Install habitat-sim with bullet physics
conda install habitat-sim withbullet -c conda-forge -c aihabitat

# Or build from source for latest features:
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install --with-cuda --bullet
```

### 3. Install Core Dependencies

```bash
# Computer vision libraries
pip install opencv-python pillow matplotlib

# Deep learning frameworks
pip install transformers==4.36.0 accelerate peft bitsandbytes

# Segmentation models
pip install ultralytics  # For FastSAM

# Additional dependencies
pip install numpy scipy networkx tqdm
pip install deepspeed  # For multi-GPU training
```

### 4. Install LLaVA-NeXT

```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
pip install -e .
```

### 5. Download Models

```bash
# Create model directory
mkdir -p models

# Download FastSAM
wget https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v1.0/FastSAM-x.pt -P models/

# Download LLaVA-OneVision
huggingface-cli download lmms-lab/llava-onevision-qwen2-0.5b-ov --local-dir models/llava-onevision-qwen2-0.5b-ov

# Download SigLIP (will auto-download on first use)
# google/siglip-so400m-patch14-384

# Download DINOv2
huggingface-cli download facebook/dinov2-base --local-dir models/dinov2-base
```

## Project Structure

```
drone-vln/
├── basic_system/              # Core navigation system
│   ├── landmark_extractor.py   # Feature extraction with FastSAM + SigLIP
│   ├── llava_navigation.py     # LLaVA-based navigation control
│   ├── navigation_system.py    # Main navigation pipeline
│   ├── r2r_test_script.py      # R2R dataset testing
│   ├── topological_graph.py    # Graph-based spatial representation
│   └── utils.py                # Utility functions
│
├── dataset_pipeline/          # Data processing & training
│   ├── drone_converter.py      # Convert to LLaVA training format
│   ├── drone_evaluate.py       # Model evaluation
│   ├── drone_finetune.py       # Fine-tuning script
│   ├── drone_frame_pairs.py    # Frame pair generation
│   └── drone_target_extractor.py # Target landmark extraction
│
├── data_collection/           # Data collection tools
│   └── drone_collector_r2r.py  # Interactive data collector
│
├── models/                    # Downloaded models
├── data/                      # Datasets
└── results/                   # Output results
```

## Quick Start

### Option 1: Test with Pre-collected Data

```bash
cd basic_system

# Run navigation test on R2R dataset
python r2r_test_script.py \
    --model_size 7b \
    --enable_visualization \
    --navigation_visualization
```

### Option 2: Full Pipeline from Scratch

#### Step 1: Collect Navigation Data

```bash
cd data_collection

# Interactive data collection (requires Habitat-Sim + MP3D scenes)
python drone_collector_r2r.py \
    --output ../data/drone_navigation_data \
    --height 1.5 \
    --port 8080

# Open http://localhost:8080 in browser to view drone feed
# Use keyboard controls: W/A/S/D/Q/E for movement, B to begin recording
```

#### Step 2: Process Dataset

```bash
cd ../dataset_pipeline

# Generate frame pairs
python drone_frame_pairs.py \
    --mode full \
    --input ../data/drone_navigation_data/annotations.json \
    --output drone_3d_frame_pairs.json

# Extract target landmarks
python drone_target_extractor.py \
    --drone_path ../data/drone_navigation_data \
    --frame_pairs drone_3d_frame_pairs.json \
    --enable_visualization

# Convert to LLaVA format
python drone_converter.py \
    --drone_targets results/drone_3d_training_targets_optimized.json \
    --drone_base_path ../data/drone_navigation_data \
    --annotations_json ../data/drone_navigation_data/annotations.json \
    --output_dir llava_drone_3d_navigation_data \
    --subset_train_size 5000 \
    --subset_val_size 1000 \
    --create_balanced
```

#### Step 3: Fine-tune Model

```bash
# Single GPU
python drone_finetune.py \
    --model_name_or_path models/llava-onevision-qwen2-7b-ov \
    --data_path llava_drone_3d_navigation_data/drone_3d_navigation_train.json \
    --image_folder llava_drone_3d_navigation_data \
    --output_dir drone_navigation_finetuned \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --bf16 True \
    --lora_enable True \
    --lora_r 128

# Multi-GPU (4 GPUs)
deepspeed --num_gpus=4 drone_finetune.py \
    --deepspeed deepspeed_config_drone_3d_4gpu.json \
    [... same args as above ...]
```

#### Step 4: Evaluate

```bash
python drone_evaluate.py \
    --model_path drone_navigation_finetuned \
    --test_data llava_drone_3d_navigation_data/drone_3d_navigation_val.json \
    --output_dir evaluation_results \
    --gpu_id 0
```

## Usage Guide


### Data Collection

The data collector provides an interactive interface for gathering navigation trajectories:

**Controls:**
- `W` - Move forward 0.25m
- `A/D` - Turn left/right 15°
- `Q/E` - Move up/down 0.25m
- `B` - Begin recording trajectory
- `S` - Stop and save trajectory
- `R` - Reset current trajectory
- `N` - Reset position
- `ESC` - Exit

**Output Format:** R2R-compatible JSON with action ground truth for each waypoint.

### Training Pipeline

The training pipeline supports:
- **3D Action Space**: 7 actions (start, forward, turn_left, turn_right, move_up, move_down, stop)
- **Dual-Target Navigation**: Visible guide + final target
- **LoRA Fine-tuning**: Efficient parameter updates
- **Multi-GPU Training**: DeepSpeed ZeRO-3 optimization

### Evaluation Metrics

- Overall navigation accuracy
- Per-action accuracy (including 3D-specific actions)
- Inference latency

## Configuration

### Key Parameters

**Navigation System:**
```python
base_similarity_threshold = 0.75  # Landmark matching threshold
quality_threshold = 0.65          # Minimum landmark quality
merge_temporal_window = 10        # Temporal merging window
```

**Data Collection:**
```python
sensor_height = 1.5               # Camera height (meters)
move_amount = 0.25                # Forward step size
turn_amount = 15.0                # Turn angle (degrees)
vertical_amount = 0.25            # Vertical step size
```

**Training:**
```python
learning_rate = 2e-5              # LoRA learning rate
lora_r = 128                      # LoRA rank
num_train_epochs = 3              # Training epochs
per_device_batch_size = 1         # Batch size per GPU
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) for vision-language model
- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) for segmentation
- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) for simulation
- [R2R Dataset](https://bringmeaspoon.org/) for navigation benchmarks
