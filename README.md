# Face GAN

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) for generating face images.

## Setup and Requirements

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- tqdm
- PIL

Install dependencies:
```bash
pip install torch torchvision matplotlib tqdm pillow
```

## Dataset Preparation

The project supports two dataset structures:

### 1. CelebDF Structure
- All images in a single folder
- Filenames follow pattern like `00012_face_699` where:
  - `00012` is the face identity number
  - `699` is the image number

### 2. FaceForensics++ Structure
- Images organized in numbered folders
- Each folder contains multiple images of the same face/identity

### Preparing Your Dataset

Use the provided utility script to prepare your dataset:

```bash
# For CelebDF dataset structure:
python setup_dataset.py --source /path/to/celebdf/faces --dataset_type celebdf --size 128

# For FaceForensics++ dataset structure:
python setup_dataset.py --source /path/to/faceforensics/faces --dataset_type faceforensics --size 128

# Additional options:
# --max_faces 100       # Limit number of unique face identities
# --max_per_face 10     # Limit number of images per face
# --balanced            # Ensure balanced image distribution across faces
```

The script will:
1. Create a `faces` directory in the project root
2. Process the source images according to the dataset structure
3. Resize all images to the specified size (default 128x128)
4. Save the processed images in the target directory

## Training

Run the main script:
```bash
python extracted_code.py
```

The script will:
1. Load images from the `faces` directory
2. Train a GAN model to generate similar face images
3. Save progress images and a final animation

## Model Architecture

- Generator: Creates 128x128 RGB images from 100-dimensional random noise
- Discriminator: Classifies 128x128 images as real or fake

## Outputs

The training process will generate several outputs:
- `real_samples.png`: Examples of real images from the dataset
- `generated_e{epoch}_i{iteration}.png`: Generated images at various stages
- `loss_curves.png`: Training loss over time
- `final_comparison.png`: Side-by-side comparison of real and generated images
- `face_gan_training.gif`: Animation showing generator progress
- `generator.pth` and `discriminator.pth`: Saved model weights

## Customization

You can modify the hyperparameters in the script:
- `BATCH_SIZE`: Number of images per batch
- `EPOCH_NUM`: Number of training epochs
- `X_DIM`: Target image size (default 128)
- `lr`: Learning rate
