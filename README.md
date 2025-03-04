
# Face GAN - Deep Convolutional GAN for Face Generation

![Sample Generated Faces](output/generated_samples.png)

## Overview

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) for generating realistic human face images. The system processes and combines data from two face datasets (CelebDF and FaceForensics++), then trains a GAN to generate new realistic faces.

## Features

- **Dual dataset processing**: Handles both CelebDF and FaceForensics++ formats
- **Combined dataset creation**: Merges multiple datasets for better training
- **Advanced GAN architecture**: Deep convolutional network optimized for face generation
- **Checkpoint system**: Save and resume training from checkpoints
- **Early stopping**: Prevents overfitting and saves time
- **Progressive visualization**: Track model improvement over time
- **Latent space exploration**: Interpolation between faces in latent space
- **Resource monitoring**: Tracks CPU, GPU, and memory usage
- **TensorBoard integration**: Visualize training progress and metrics

## Project Structure

- `face_gan_enhanced.py` - Main script with complete pipeline
- `face_gan_notebook_sections.py` - Code sections for Jupyter notebook
- `processed_faces/` - Directory containing processed face datasets
  - `celebdf/` - Processed CelebDF dataset
  - `faceforensics/` - Processed FaceForensics++ dataset
  - `combined/` - Combined dataset used for training
- `output/` - Output directory for models and visualizations
  - `checkpoints/` - Saved model checkpoints
  - `logs/` - Training log files and TensorBoard logs

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- matplotlib
- Pillow (PIL)
- tqdm
- numpy
- (Optional) tensorboard
- (Optional) psutil and GPUtil for resource monitoring

## Usage

### Option 1: Running the Full Script

