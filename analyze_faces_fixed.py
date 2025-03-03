import os
import argparse
import subprocess
from pathlib import Path

def main():
    """
    Run analysis on specific CelebDF and FaceForensics++ datasets
    with proper Windows path formatting using double backslashes
    """
    parser = argparse.ArgumentParser(description='Analyze face datasets from specific paths')
    parser.add_argument('--analyze_celebdf', action='store_true', help='Analyze CelebDF dataset')
    parser.add_argument('--analyze_ff', action='store_true', help='Analyze FaceForensics++ dataset')
    parser.add_argument('--process', action='store_true', help='Process datasets after analysis')
    parser.add_argument('--max_per_face', type=int, default=None, help='Maximum images per face')
    parser.add_argument('--max_faces', type=int, default=None, help='Maximum number of faces')
    parser.add_argument('--size', type=int, default=128, help='Target image size')
    parser.add_argument('--balanced', action='store_true', help='Create balanced dataset')
    parser.add_argument('--target', type=str, default='.\\processed_faces', 
                       help='Target directory for processed faces')
    
    args = parser.parse_args()
    
    # Get the base directory (mnist folder)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to the mnist folder
    celebdf_path = os.path.join(base_dir, "faces\\Real\\Celeb_V2\\Train\\real")
    ff_path = os.path.join(base_dir, "faces\\Real\\FaceForensics++\\original_sequences\\youtube\\c23\\frames")
    
    # For display purposes, show only paths relative to mnist folder
    celebdf_display_path = "faces\\Real\\Celeb_V2\\Train\\real"
    ff_display_path = "faces\\Real\\FaceForensics++\\original_sequences\\youtube\\c23\\frames"
    
    print(f"CelebDF dataset path: {celebdf_display_path}")
    print(f"FaceForensics++ dataset path: {ff_display_path}")
    
    if not os.path.exists(celebdf_path) and args.analyze_celebdf:
        print(f"Error: CelebDF path does not exist: {celebdf_display_path}")
        return
        
    if not os.path.exists(ff_path) and args.analyze_ff:
        print(f"Error: FaceForensics++ path does not exist: {ff_display_path}")
        return
    
    # Create output directory for reports
    reports_dir = Path(".\\dataset_reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Create target directory if processing
    if args.process:
        target_path = Path(args.target)
        target_path.mkdir(exist_ok=True)
    
    # Analyze CelebDF
    if args.analyze_celebdf:
        print("\n" + "="*50)
        print("ANALYZING CELEBDF DATASET")
        print("="*50)
        
        cmd = [
            "python", "dataset_summary.py",
            "--source", celebdf_path,
            "--dataset_type", "celebdf"
        ]
        
        if args.process:
            cmd.extend([
                "--process",
                "--target", os.path.join(args.target, "celebdf").replace('/', '\\')
            ])
            
            if args.max_per_face:
                cmd.extend(["--max_per_face", str(args.max_per_face)])
                
            if args.max_faces:
                cmd.extend(["--max_faces", str(args.max_faces)])
                
            if args.size:
                cmd.extend(["--size", str(args.size)])
        
        # Display command with paths hidden
        display_cmd = [
            "python", "dataset_summary.py",
            "--source", celebdf_display_path,
            "--dataset_type", "celebdf"
        ]
        
        if args.process:
            display_cmd.extend([
                "--process",
                "--target", os.path.join(args.target, "celebdf").replace('/', '\\')
            ])
            
            if args.max_per_face:
                display_cmd.extend(["--max_per_face", str(args.max_per_face)])
                
            if args.max_faces:
                display_cmd.extend(["--max_faces", str(args.max_faces)])
                
            if args.size:
                display_cmd.extend(["--size", str(args.size)])
                
        print(f"Running: {' '.join(display_cmd)}")
        subprocess.run(cmd)
        
        # Rename output files for clarity
        if os.path.exists("celebdf_distribution.png"):
            os.rename("celebdf_distribution.png", 
                     os.path.join(reports_dir, "celebdf_distribution.png").replace('/', '\\'))
        
        if os.path.exists("celebdf_samples.png"):
            os.rename("celebdf_samples.png", 
                     os.path.join(reports_dir, "celebdf_samples.png").replace('/', '\\'))
            
        if os.path.exists("processed_distribution.png"):
            os.rename("processed_distribution.png", 
                     os.path.join(reports_dir, "celebdf_processed_distribution.png").replace('/', '\\'))
            
        if os.path.exists("dataset_report.txt"):
            with open("dataset_report.txt", "r") as f:
                celebdf_report = f.read()
            
            # Replace full paths with relative paths in the report
            celebdf_report = celebdf_report.replace(celebdf_path, celebdf_display_path)
            
            with open(os.path.join(reports_dir, "celebdf_report.txt").replace('/', '\\'), "w") as f:
                f.write(celebdf_report)
    
    # Analyze FaceForensics++
    if args.analyze_ff:
        print("\n" + "="*50)
        print("ANALYZING FACEFORENSICS++ DATASET")
        print("="*50)
        
        cmd = [
            "python", "dataset_summary.py",
            "--source", ff_path,
            "--dataset_type", "faceforensics"
        ]
        
        if args.process:
            cmd.extend([
                "--process",
                "--target", os.path.join(args.target, "faceforensics").replace('/', '\\')
            ])
            
            if args.max_per_face:
                cmd.extend(["--max_per_face", str(args.max_per_face)])
                
            if args.max_faces:
                cmd.extend(["--max_faces", str(args.max_faces)])
                
            if args.size:
                cmd.extend(["--size", str(args.size)])
        
        # Display command with paths hidden
        display_cmd = [
            "python", "dataset_summary.py",
            "--source", ff_display_path,
            "--dataset_type", "faceforensics"
        ]
        
        if args.process:
            display_cmd.extend([
                "--process",
                "--target", os.path.join(args.target, "faceforensics").replace('/', '\\')
            ])
            
            if args.max_per_face:
                display_cmd.extend(["--max_per_face", str(args.max_per_face)])
                
            if args.max_faces:
                display_cmd.extend(["--max_faces", str(args.max_faces)])
                
            if args.size:
                display_cmd.extend(["--size", str(args.size)])
        
        print(f"Running: {' '.join(display_cmd)}")
        subprocess.run(cmd)
        
        # Rename output files for clarity
        if os.path.exists("faceforensics_distribution.png"):
            os.rename("faceforensics_distribution.png", 
                     os.path.join(reports_dir, "faceforensics_distribution.png").replace('/', '\\'))
        
        if os.path.exists("faceforensics_samples.png"):
            os.rename("faceforensics_samples.png", 
                     os.path.join(reports_dir, "faceforensics_samples.png").replace('/', '\\'))
            
        if os.path.exists("processed_distribution.png"):
            os.rename("processed_distribution.png", 
                     os.path.join(reports_dir, "faceforensics_processed_distribution.png").replace('/', '\\'))
            
        if os.path.exists("dataset_report.txt"):
            with open("dataset_report.txt", "r") as f:
                ff_report = f.read()
            
            # Replace full paths with relative paths in the report
            ff_report = ff_report.replace(ff_path, ff_display_path)
            
            with open(os.path.join(reports_dir, "faceforensics_report.txt").replace('/', '\\'), "w") as f:
                f.write(ff_report)
    
    # Generate combined report if both were analyzed
    if args.analyze_celebdf and args.analyze_ff:
        try:
            with open(os.path.join(reports_dir, "celebdf_report.txt").replace('/', '\\'), "r") as f:
                celebdf_report = f.read()
                
            with open(os.path.join(reports_dir, "faceforensics_report.txt").replace('/', '\\'), "r") as f:
                ff_report = f.read()
                
            with open(os.path.join(reports_dir, "combined_report.txt").replace('/', '\\'), "w") as f:
                f.write("="*50 + "\n")
                f.write("COMBINED DATASET ANALYSIS REPORT\n")
                f.write("="*50 + "\n\n")
                
                f.write("-"*50 + "\n")
                f.write("CELEBDF REPORT\n")
                f.write("-"*50 + "\n\n")
                f.write(celebdf_report)
                
                f.write("\n\n" + "-"*50 + "\n")
                f.write("FACEFORENSICS++ REPORT\n")
                f.write("-"*50 + "\n\n")
                f.write(ff_report)
                
                if args.process:
                    f.write("\n\n" + "="*50 + "\n")
                    f.write("TRAINING PREPARATION\n")
                    f.write("="*50 + "\n\n")
                    f.write("To use the processed datasets for GAN training:\n\n")
                    f.write("1. Individual datasets:\n")
                    f.write(f"   - CelebDF: {os.path.join(args.target, 'celebdf').replace('/', '\\')}\n")
                    f.write(f"   - FaceForensics++: {os.path.join(args.target, 'faceforensics').replace('/', '\\')}\n\n")
                    f.write("2. Combined dataset (requires manual copying):\n")
                    f.write(f"   - Copy all images from both processed folders to: {os.path.join(args.target, 'combined').replace('/', '\\')}\n")
                    f.write("   - Update DATASET_PATH in extracted_code.py to point to the combined folder\n")
            
            print(f"\nCombined report generated: {os.path.join(reports_dir, 'combined_report.txt').replace('/', '\\')}")
            
        except Exception as e:
            print(f"Error generating combined report: {e}")
    
    # Create combined dataset if both were processed
    if args.analyze_celebdf and args.analyze_ff and args.process:
        combined_dir = os.path.join(args.target, "combined").replace('/', '\\')
        os.makedirs(combined_dir, exist_ok=True)
        
        print("\n" + "="*50)
        print("CREATING COMBINED DATASET")
        print("="*50)
        
        # Create a script to combine the datasets
        combine_script = os.path.join(reports_dir, "combine_datasets.py").replace('/', '\\')
        with open(combine_script, "w") as f:
            f.write("# Script to combine processed datasets\n")
            f.write("import os\n")
            f.write("import shutil\n")
            f.write("from pathlib import Path\n\n")
            
            celebdf_dir_path = os.path.join(args.target, 'celebdf').replace('/', '\\')
            ff_dir_path = os.path.join(args.target, 'faceforensics').replace('/', '\\')
            
            f.write(f"celebdf_dir = Path(r'{celebdf_dir_path}')\n")
            f.write(f"ff_dir = Path(r'{ff_dir_path}')\n")
            f.write(f"combined_dir = Path(r'{combined_dir}')\n\n")
            
            f.write("combined_dir.mkdir(exist_ok=True)\n\n")
            
            f.write("# Copy CelebDF images\n")
            f.write("celebdf_files = list(celebdf_dir.glob('*.jpg')) + list(celebdf_dir.glob('*.png'))\n")
            f.write("print(f'Copying {len(celebdf_files)} images from CelebDF')\n")
            f.write("for i, src_file in enumerate(celebdf_files):\n")
            f.write("    dst_file = combined_dir / f'celebdf_{i:05d}{src_file.suffix}'\n")
            f.write("    shutil.copy2(src_file, dst_file)\n")
            f.write("    if i % 100 == 0:\n")
            f.write("        print(f'Copied {i} CelebDF images')\n\n")
            
            f.write("# Copy FaceForensics++ images\n")
            f.write("ff_files = list(ff_dir.glob('*.jpg')) + list(ff_dir.glob('*.png'))\n")
            f.write("print(f'Copying {len(ff_files)} images from FaceForensics++')\n")
            f.write("for i, src_file in enumerate(ff_files):\n")
            f.write("    dst_file = combined_dir / f'ff_{i:05d}{src_file.suffix}'\n")
            f.write("    shutil.copy2(src_file, dst_file)\n")
            f.write("    if i % 100 == 0:\n")
            f.write("        print(f'Copied {i} FaceForensics++ images')\n\n")
            
            f.write("print(f'Combined dataset created at {combined_dir}')\n")
            f.write("print(f'Total images: {len(celebdf_files) + len(ff_files)}')\n")
        
        print(f"Created combine script: {combine_script}")
        print("To combine datasets, run:")
        print(f"python {combine_script}")
    
    print("\nAnalysis complete!")
    print(f"Reports saved in: {reports_dir}")
    if args.process:
        print(f"Processed datasets saved in: {args.target}")
        print("\nTo train GAN with processed dataset, update DATASET_PATH in extracted_code.py:")
        if args.analyze_celebdf and args.analyze_ff:
            print(f"DATASET_PATH = '.\\\\processed_faces\\\\combined'  # Combined dataset")
        elif args.analyze_celebdf:
            print(f"DATASET_PATH = '.\\\\processed_faces\\\\celebdf'  # CelebDF dataset")
        elif args.analyze_ff:
            print(f"DATASET_PATH = '.\\\\processed_faces\\\\faceforensics'  # FaceForensics++ dataset")

if __name__ == "__main__":
    main()
