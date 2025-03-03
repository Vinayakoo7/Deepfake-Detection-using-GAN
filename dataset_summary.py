import os
import re
from pathlib import Path
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random

# Import functions from setup_dataset.py
from setup_dataset import create_faces_directory, process_celebdf_dataset, process_faceforensics_dataset

def analyze_celebdf_dataset(source_dir):
    """
    Analyze CelebDF dataset structure and return statistics
    
    Args:
        source_dir: Path to the CelebDF dataset directory
    
    Returns:
        dict: Statistics about the dataset
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_path} does not exist.")
        return None
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(source_path.glob(f"*{ext}"))
        image_files.extend(source_path.glob(f"*{ext.upper()}"))
    
    # Parse face IDs from filenames
    face_pattern = re.compile(r'^(\d+)_face_\d+')
    face_images = defaultdict(list)
    
    for img_path in image_files:
        match = face_pattern.match(img_path.stem)
        if match:
            face_id = match.group(1)
            face_images[face_id].append(img_path)
    
    # Calculate statistics
    total_images = len(image_files)
    unique_faces = len(face_images)
    
    # Calculate distribution of images per face
    images_per_face = [len(images) for face_id, images in face_images.items()]
    min_images = min(images_per_face) if images_per_face else 0
    max_images = max(images_per_face) if images_per_face else 0
    avg_images = sum(images_per_face) / len(images_per_face) if images_per_face else 0
    
    return {
        'total_images': total_images,
        'unique_faces': unique_faces,
        'min_images_per_face': min_images,
        'max_images_per_face': max_images,
        'avg_images_per_face': avg_images,
        'face_distribution': images_per_face
    }

def analyze_faceforensics_dataset(source_dir):
    """
    Analyze FaceForensics++ dataset structure and return statistics
    
    Args:
        source_dir: Path to the FaceForensics++ dataset directory
    
    Returns:
        dict: Statistics about the dataset
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_path} does not exist.")
        return None
    
    # Get all folders in the source directory
    face_folders = [f for f in source_path.iterdir() if f.is_dir()]
    
    # Count images in each folder
    folder_counts = []
    total_images = 0
    
    for folder in face_folders:
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))
        
        folder_counts.append(len(image_files))
        total_images += len(image_files)
    
    # Calculate statistics
    unique_faces = len(face_folders)
    min_images = min(folder_counts) if folder_counts else 0
    max_images = max(folder_counts) if folder_counts else 0
    avg_images = sum(folder_counts) / len(folder_counts) if folder_counts else 0
    
    return {
        'total_images': total_images,
        'unique_faces': unique_faces,
        'min_images_per_face': min_images,
        'max_images_per_face': max_images,
        'avg_images_per_face': avg_images,
        'face_distribution': folder_counts
    }

def analyze_processed_dataset(target_dir):
    """
    Analyze the processed dataset in the target directory
    
    Args:
        target_dir: Path to the processed dataset directory
    
    Returns:
        dict: Statistics about the dataset
    """
    target_path = Path(target_dir)
    
    if not target_path.exists():
        print(f"Target directory {target_path} does not exist.")
        return None
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(target_path.glob(f"*{ext}"))
        image_files.extend(target_path.glob(f"*{ext.upper()}"))
    
    # Parse face IDs from filenames
    face_pattern = re.compile(r'^face_(\d+)_\d+')
    face_images = defaultdict(list)
    
    for img_path in image_files:
        match = face_pattern.match(img_path.stem)
        if match:
            face_id = match.group(1)
            face_images[face_id].append(img_path)
    
    # Calculate statistics
    total_images = len(image_files)
    unique_faces = len(face_images)
    
    # Calculate distribution of images per face
    images_per_face = [len(images) for face_id, images in face_images.items()]
    min_images = min(images_per_face) if images_per_face else 0
    max_images = max(images_per_face) if images_per_face else 0
    avg_images = sum(images_per_face) / len(images_per_face) if images_per_face else 0
    
    return {
        'total_images': total_images,
        'unique_faces': unique_faces,
        'min_images_per_face': min_images,
        'max_images_per_face': max_images,
        'avg_images_per_face': avg_images,
        'face_distribution': images_per_face
    }

def plot_face_distribution(stats, title, output_file):
    """Plot the distribution of images per face"""
    plt.figure(figsize=(10, 6))
    plt.hist(stats['face_distribution'], bins=30)
    plt.title(f"{title} - Distribution of Images per Face")
    plt.xlabel("Number of Images")
    plt.ylabel("Number of Faces")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file)
    plt.close()

def sample_faces_plot(source_dir, dataset_type, output_file, samples=5, images_per_face=3):
    """Create a plot showing sample faces from the dataset"""
    source_path = Path(source_dir)
    
    if dataset_type == 'celebdf':
        # Get face groupings
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_path.glob(f"*{ext}"))
        
        face_pattern = re.compile(r'^(\d+)_face_\d+')
        face_images = defaultdict(list)
        
        for img_path in image_files:
            match = face_pattern.match(img_path.stem)
            if match:
                face_id = match.group(1)
                face_images[face_id].append(img_path)
        
        # Sample faces and images
        face_ids = list(face_images.keys())
        if len(face_ids) > samples:
            selected_faces = random.sample(face_ids, samples)
        else:
            selected_faces = face_ids
    
    else:  # faceforensics
        # Get folders
        face_folders = [f for f in source_path.iterdir() if f.is_dir()]
        
        # Sample folders
        if len(face_folders) > samples:
            selected_folders = random.sample(face_folders, samples)
        else:
            selected_folders = face_folders
        
        # Create face_images dictionary
        face_images = {}
        for i, folder in enumerate(selected_folders):
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png']:
                image_files.extend(folder.glob(f"*{ext}"))
            face_images[str(i)] = image_files
        
        selected_faces = list(face_images.keys())
    
    # Create plot
    fig, axes = plt.subplots(samples, images_per_face, figsize=(images_per_face*3, samples*3))
    
    # If only one face, convert axes to 2D array
    if samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, face_id in enumerate(selected_faces):
        images = face_images[face_id]
        if len(images) > images_per_face:
            selected_images = random.sample(images, images_per_face)
        else:
            selected_images = images[:images_per_face]
        
        # Fill remaining slots with blank images if needed
        while len(selected_images) < images_per_face:
            selected_images.append(None)
        
        for j, img_path in enumerate(selected_images):
            if img_path is None:
                axes[i, j].axis('off')
                continue
                
            try:
                img = Image.open(img_path)
                axes[i, j].imshow(np.array(img))
                axes[i, j].set_title(f"Face {face_id}")
                axes[i, j].axis('off')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def print_summary(stats, dataset_name):
    """Print a summary of the dataset statistics"""
    if stats is None:
        print(f"No statistics available for {dataset_name} dataset.")
        return
    
    print("\n" + "="*50)
    print(f"SUMMARY FOR {dataset_name.upper()} DATASET")
    print("="*50)
    print(f"Total images: {stats['total_images']}")
    print(f"Unique faces: {stats['unique_faces']}")
    print(f"Images per face:")
    print(f"  - Minimum: {stats['min_images_per_face']}")
    print(f"  - Maximum: {stats['max_images_per_face']}")
    print(f"  - Average: {stats['avg_images_per_face']:.2f}")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze face dataset statistics')
    parser.add_argument('--source', type=str, help='Source directory containing face images')
    parser.add_argument('--dataset_type', type=str, choices=['celebdf', 'faceforensics'], 
                        help='Type of dataset structure')
    parser.add_argument('--target', type=str, default='./faces', 
                        help='Target directory for processed images (default: ./faces)')
    parser.add_argument('--process', action='store_true',
                        help='Process the dataset into the target directory')
    parser.add_argument('--size', type=int, default=128, 
                        help='Target size for processed images (default: 128)')
    parser.add_argument('--max_faces', type=int, default=None, 
                        help='Maximum number of unique faces to include')
    parser.add_argument('--max_per_face', type=int, default=None,
                        help='Maximum number of images per face')
    
    args = parser.parse_args()
    
    if not args.source:
        print("No source directory specified. Please provide a source directory with --source.")
        return
    
    if not args.dataset_type:
        print("No dataset type specified. Please provide --dataset_type (celebdf or faceforensics).")
        return
    
    # Analyze source dataset
    if args.dataset_type == 'celebdf':
        source_stats = analyze_celebdf_dataset(args.source)
    else:  # faceforensics
        source_stats = analyze_faceforensics_dataset(args.source)
    
    print_summary(source_stats, f"Source {args.dataset_type}")
    
    if source_stats:
        # Plot distribution
        plot_face_distribution(source_stats, f"{args.dataset_type.upper()} Dataset", f"{args.dataset_type}_distribution.png")
        
        # Plot sample faces
        sample_faces_plot(args.source, args.dataset_type, f"{args.dataset_type}_samples.png")
    
    # Process dataset if requested
    if args.process:
        target_dir = Path(args.target)
        target_dir.mkdir(exist_ok=True)
        
        if args.dataset_type == 'celebdf':
            process_celebdf_dataset(args.source, target_dir, (args.size, args.size), 
                                   max_images_per_face=args.max_per_face)
        else:  # faceforensics
            process_faceforensics_dataset(args.source, target_dir, (args.size, args.size),
                                         max_folders=args.max_faces,
                                         max_images_per_folder=args.max_per_face)
        
        # Analyze processed dataset
        processed_stats = analyze_processed_dataset(target_dir)
        print_summary(processed_stats, "Processed")
        
        if processed_stats:
            # Plot distribution
            plot_face_distribution(processed_stats, "Processed Dataset", "processed_distribution.png")

    # Generate final report
    with open('dataset_report.txt', 'w') as f:
        f.write("="*50 + "\n")
        f.write("FACE DATASET ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Dataset type: {args.dataset_type.upper()}\n")
        f.write(f"Source directory: {args.source}\n")
        
        if source_stats:
            f.write("\n" + "-"*50 + "\n")
            f.write("SOURCE DATASET STATISTICS\n")
            f.write("-"*50 + "\n")
            f.write(f"Total images: {source_stats['total_images']}\n")
            f.write(f"Unique faces: {source_stats['unique_faces']}\n")
            f.write(f"Images per face:\n")
            f.write(f"  - Minimum: {source_stats['min_images_per_face']}\n")
            f.write(f"  - Maximum: {source_stats['max_images_per_face']}\n")
            f.write(f"  - Average: {source_stats['avg_images_per_face']:.2f}\n")
        
        if args.process and processed_stats:
            f.write("\n" + "-"*50 + "\n")
            f.write("PROCESSED DATASET STATISTICS\n")
            f.write("-"*50 + "\n")
            f.write(f"Target directory: {args.target}\n")
            f.write(f"Image size: {args.size}x{args.size}\n")
            f.write(f"Total images: {processed_stats['total_images']}\n")
            f.write(f"Unique faces: {processed_stats['unique_faces']}\n")
            f.write(f"Images per face:\n")
            f.write(f"  - Minimum: {processed_stats['min_images_per_face']}\n")
            f.write(f"  - Maximum: {processed_stats['max_images_per_face']}\n")
            f.write(f"  - Average: {processed_stats['avg_images_per_face']:.2f}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write("TRAINING IMPLICATIONS\n")
        f.write("="*50 + "\n")
        f.write("The GAN model training treats each image independently and does not\n")
        f.write("explicitly track or use face identity information during training.\n")
        f.write("However, the distribution of faces and number of images per face\n")
        f.write("will implicitly affect what the model learns.\n\n")
        f.write("Faces with more images will have greater influence on the training.\n")
        f.write("For more balanced training, consider using the --max_per_face option\n")
        f.write("when processing the dataset.\n")
    
    print(f"\nReport generated: dataset_report.txt")
    if source_stats:
        print(f"Distribution plot saved: {args.dataset_type}_distribution.png")
        print(f"Sample faces plot saved: {args.dataset_type}_samples.png")
    if args.process and processed_stats:
        print(f"Processed distribution plot saved: processed_distribution.png")

if __name__ == "__main__":
    main()
