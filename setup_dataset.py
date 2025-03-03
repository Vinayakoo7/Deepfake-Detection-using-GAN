import os
import shutil
import re
from pathlib import Path
import argparse
from PIL import Image
import numpy as np
import random

def create_faces_directory():
    """Create the faces directory if it doesn't exist"""
    face_dir = Path("./faces")
    face_dir.mkdir(exist_ok=True)
    return face_dir

def process_celebdf_dataset(source_dir, target_dir, target_size=(128, 128), max_images_per_face=None):
    """
    Process CelebDF dataset where all images are in a single folder
    with naming format like 00012_face_699 (face_id_face_image_number)
    
    Args:
        source_dir: Path to the CelebDF dataset directory
        target_dir: Path to save the processed images
        target_size: Size to resize images to
        max_images_per_face: Maximum number of images to use per unique face identity
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_path} does not exist.")
        return 0
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(source_path.glob(f"*{ext}"))
        image_files.extend(source_path.glob(f"*{ext.upper()}"))
    
    # Parse face IDs from filenames
    face_pattern = re.compile(r'^(\d+)_face_\d+')
    face_images = {}
    
    for img_path in image_files:
        match = face_pattern.match(img_path.stem)
        if match:
            face_id = match.group(1)
            if face_id not in face_images:
                face_images[face_id] = []
            face_images[face_id].append(img_path)
    
    print(f"Found {len(image_files)} total images across {len(face_images)} unique faces in {source_path}")
    
    # Process images by face
    count = 0
    for face_id, images in face_images.items():
        # Limit number of images per face if specified
        if max_images_per_face and len(images) > max_images_per_face:
            images = random.sample(images, max_images_per_face)
        
        for img_path in images:
            try:
                # Open and resize image
                img = Image.open(img_path)
                img = img.resize(target_size, Image.LANCZOS)
                
                # Save to target directory
                target_file = target_path / f"face_{face_id}_{count:04d}{img_path.suffix}"
                img.save(target_file)
                count += 1
                
                # Print progress every 100 images
                if count % 100 == 0:
                    print(f"Processed {count} images")
                    
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
    
    print(f"Successfully processed {count} images to {target_path}")
    return count

def process_faceforensics_dataset(source_dir, target_dir, target_size=(128, 128), max_folders=None, max_images_per_folder=None):
    """
    Process FaceForensics++ dataset where images are organized in numbered folders
    
    Args:
        source_dir: Path to the FaceForensics++ dataset directory
        target_dir: Path to save the processed images
        target_size: Size to resize images to
        max_folders: Maximum number of person/face folders to process
        max_images_per_folder: Maximum number of images to use per folder
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_path} does not exist.")
        return 0
    
    # Get all folders in the source directory
    face_folders = [f for f in source_path.iterdir() if f.is_dir()]
    
    # Limit number of folders if specified
    if max_folders and len(face_folders) > max_folders:
        face_folders = random.sample(face_folders, max_folders)
    
    print(f"Found {len(face_folders)} face folders in {source_path}")
    
    # Process images in each folder
    count = 0
    for folder_idx, folder in enumerate(face_folders):
        # Get image files in this folder
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))
        
        # Limit number of images per folder if specified
        if max_images_per_folder and len(image_files) > max_images_per_folder:
            image_files = random.sample(image_files, max_images_per_folder)
        
        for img_idx, img_path in enumerate(image_files):
            try:
                # Open and resize image
                img = Image.open(img_path)
                img = img.resize(target_size, Image.LANCZOS)
                
                # Save to target directory with folder index as face ID
                target_file = target_path / f"face_{folder_idx:04d}_{img_idx:04d}{img_path.suffix}"
                img.save(target_file)
                count += 1
                
                # Print progress every 100 images
                if count % 100 == 0:
                    print(f"Processed {count} images")
                    
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
    
    print(f"Successfully processed {count} images to {target_path}")
    return count

def main():
    parser = argparse.ArgumentParser(description='Prepare face dataset for GAN training')
    parser.add_argument('--source', type=str, help='Source directory containing face images')
    parser.add_argument('--size', type=int, default=128, help='Target size for images (default: 128)')
    parser.add_argument('--dataset_type', type=str, choices=['celebdf', 'faceforensics'], 
                        help='Type of dataset structure (celebdf: all images in one folder, faceforensics: images in numbered folders)')
    parser.add_argument('--max_faces', type=int, default=None, 
                        help='Maximum number of unique faces to include (for CelebDF) or folders to process (for FaceForensics++)')
    parser.add_argument('--max_per_face', type=int, default=None,
                        help='Maximum number of images to include per unique face/folder')
    parser.add_argument('--balanced', action='store_true',
                        help='Ensure balanced distribution of images across face identities')
    
    args = parser.parse_args()
    
    # Create faces directory
    target_dir = create_faces_directory()
    
    if not args.source:
        print("No source directory specified. Please provide a source directory with --source.")
        print("Example: python setup_dataset.py --source path/to/face/images --size 128 --dataset_type celebdf")
        return
    
    if not args.dataset_type:
        print("No dataset type specified. Please provide --dataset_type (celebdf or faceforensics).")
        return
    
    # Process based on dataset type
    if args.dataset_type == 'celebdf':
        process_celebdf_dataset(args.source, target_dir, (args.size, args.size), 
                                max_images_per_face=args.max_per_face if args.balanced else None)
    else:  # faceforensics
        process_faceforensics_dataset(args.source, target_dir, (args.size, args.size),
                                     max_folders=args.max_faces,
                                     max_images_per_folder=args.max_per_face if args.balanced else None)

if __name__ == "__main__":
    main()
