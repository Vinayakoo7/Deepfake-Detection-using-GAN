import os
import argparse
import torch
import torch.nn as nn
import torchvision.utils as vutils
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
import math
import cv2

# Import face_recognition library IF installed
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    # Warning will be printed later if filter is requested

# %%
###############################################################################
# CONFIGURATION (Unchanged)
###############################################################################
IMAGE_CHANNEL = 3; Z_DIM = 100; G_HIDDEN = 128; D_HIDDEN = 128; OUTPUT_SIZE = 128

# %%
###############################################################################
# MODEL ARCHITECTURES (Generator and Discriminator - unchanged)
###############################################################################
class Generator(nn.Module):
    """Generator Network"""
    def __init__(self): super().__init__();self.initial=nn.Sequential(nn.ConvTranspose2d(Z_DIM,G_HIDDEN*16,4,1,0,bias=False),nn.BatchNorm2d(G_HIDDEN*16),nn.LeakyReLU(0.2,inplace=True));self.stage1=nn.Sequential(nn.ConvTranspose2d(G_HIDDEN*16,G_HIDDEN*8,4,2,1,bias=False),nn.BatchNorm2d(G_HIDDEN*8),nn.LeakyReLU(0.2,inplace=True));self.stage2=nn.Sequential(nn.ConvTranspose2d(G_HIDDEN*8,G_HIDDEN*4,4,2,1,bias=False),nn.BatchNorm2d(G_HIDDEN*4),nn.LeakyReLU(0.2,inplace=True));self.stage3=nn.Sequential(nn.ConvTranspose2d(G_HIDDEN*4,G_HIDDEN*2,4,2,1,bias=False),nn.BatchNorm2d(G_HIDDEN*2),nn.LeakyReLU(0.2,inplace=True));self.stage4=nn.Sequential(nn.ConvTranspose2d(G_HIDDEN*2,G_HIDDEN,4,2,1,bias=False),nn.BatchNorm2d(G_HIDDEN),nn.LeakyReLU(0.2,inplace=True));self.final=nn.Sequential(nn.ConvTranspose2d(G_HIDDEN,IMAGE_CHANNEL,4,2,1,bias=False),nn.Tanh());self.res1=self._make_residual(G_HIDDEN*8);self.res2=self._make_residual(G_HIDDEN*4);self.res3=self._make_residual(G_HIDDEN*2)
    def _make_residual(self,c): return nn.Sequential(nn.Conv2d(c,c,3,1,1,bias=False),nn.BatchNorm2d(c),nn.LeakyReLU(0.2,inplace=True),nn.Conv2d(c,c,3,1,1,bias=False),nn.BatchNorm2d(c))
    def forward(self,i): x=self.initial(i);x=self.stage1(x);r=self.res1(x);x=x+0.2*r;x=self.stage2(x);r=self.res2(x);x=x+0.2*r;x=self.stage3(x);r=self.res3(x);x=x+0.2*r;x=self.stage4(x);return self.final(x)

class Discriminator(nn.Module):
    """Discriminator Network"""
    def __init__(self): super().__init__();self.model=nn.Sequential(self._spectral_norm(nn.Conv2d(IMAGE_CHANNEL,D_HIDDEN,4,2,1,bias=False)),nn.LeakyReLU(0.2,inplace=True),self._spectral_norm(nn.Conv2d(D_HIDDEN,D_HIDDEN*2,4,2,1,bias=False)),nn.BatchNorm2d(D_HIDDEN*2),nn.LeakyReLU(0.2,inplace=True),self._spectral_norm(nn.Conv2d(D_HIDDEN*2,D_HIDDEN*4,4,2,1,bias=False)),nn.BatchNorm2d(D_HIDDEN*4),nn.LeakyReLU(0.2,inplace=True),self._spectral_norm(nn.Conv2d(D_HIDDEN*4,D_HIDDEN*8,4,2,1,bias=False)),nn.BatchNorm2d(D_HIDDEN*8),nn.LeakyReLU(0.2,inplace=True),self._spectral_norm(nn.Conv2d(D_HIDDEN*8,D_HIDDEN*16,4,2,1,bias=False)),nn.BatchNorm2d(D_HIDDEN*16),nn.LeakyReLU(0.2,inplace=True),self._spectral_norm(nn.Conv2d(D_HIDDEN*16,1,4,1,0,bias=False)),nn.Sigmoid())
    def _spectral_norm(self,m): return nn.utils.spectral_norm(m)
    def forward(self,i): return self.model(i).view(-1)

# %%
###############################################################################
# HELPER FUNCTIONS FOR IMAGE CONVERSION & FILTERS
###############################################################################

def tensor_to_rgb_numpy(tensor_img):
    """Converts tensor (C, H, W) [-1, 1] to NumPy RGB [0, 255]."""
    img = torch.clamp(tensor_img * 0.5 + 0.5, 0, 1)
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    img_rgb = (img_np * 255).astype(np.uint8)
    return np.ascontiguousarray(img_rgb)

def detect_faces_face_recognition(rgb_numpy_img):
    """Detects faces using face_recognition library."""
    if not FACE_RECOGNITION_AVAILABLE: return []
    return face_recognition.face_locations(rgb_numpy_img, model="cnn")

def get_face_landmarks(rgb_numpy_img, face_locations):
     """Gets landmarks for detected faces."""
     if not FACE_RECOGNITION_AVAILABLE: return []
     # Provide face_locations to speed up landmark finding
     return face_recognition.face_landmarks(rgb_numpy_img, face_locations=face_locations)

def validate_landmarks(landmark_list):
    """Performs basic checks on detected landmarks."""
    if not landmark_list: # No landmarks detected for any face
        return False

    # Expecting landmarks for exactly one face
    if len(landmark_list) != 1:
         return False # Should have been caught by face count filter, but double check

    landmarks = landmark_list[0] # Get landmarks for the first (only) face

    # Check presence of essential features
    required_features = ['left_eye', 'right_eye', 'nose_bridge', 'top_lip', 'bottom_lip']
    if not all(feature in landmarks for feature in required_features):
        return False

    # --- Optional: Add more sophisticated checks here if needed ---
    # Example: Basic vertical ordering (simplified)
    try:
        eye_y = (np.mean([p[1] for p in landmarks['left_eye']]) + np.mean([p[1] for p in landmarks['right_eye']])) / 2
        nose_y = np.mean([p[1] for p in landmarks['nose_bridge']])
        mouth_y = (np.mean([p[1] for p in landmarks['top_lip']]) + np.mean([p[1] for p in landmarks['bottom_lip']])) / 2
        if not (eye_y < nose_y < mouth_y):
            # print(" Landmark vertical order check failed") # Uncomment for debugging
            return False
    except (IndexError, KeyError, ValueError):
        # Handle cases where landmarks might be incomplete or calculation fails
        return False # Treat calculation failure as invalid

    return True # Passed basic checks

def calculate_sharpness(rgb_numpy_img):
    """Calculates the variance of the Laplacian for sharpness."""
    if rgb_numpy_img is None or rgb_numpy_img.size == 0: return 0.0
    try:
        gray = cv2.cvtColor(rgb_numpy_img, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    except: return 0.0 # Catch potential errors during conversion/calculation

# %%
###############################################################################
# MAIN EXECUTION BLOCK
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Images using a Trained GAN with Optional Filters")
    # --- Core Arguments ---
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the combined training checkpoint file (.pt).")
    parser.add_argument("--num_images", type=int, required=True, help="Number of images that meet ALL criteria to generate and save")
    parser.add_argument("--output_dir", type=str, default="./generated_images", help="Directory to save the final generated images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation (face detect/landmarks can be slow)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use ('auto', 'cuda', 'cpu')")
    parser.add_argument('--seed', type=int, default=None, help='Random seed')

    # --- GAN Filtering Arguments ---
    parser.add_argument("--truncation_psi", type=float, default=1.0, help="Truncation psi value (0.0 to 1.0).")
    parser.add_argument("--filter_d", action='store_true', help="Enable filtering using the Discriminator score.")
    parser.add_argument("--filter_d_threshold", type=float, default=0.3, help="Discriminator score threshold.")

    # --- Post-Processing Filtering Arguments ---
    parser.add_argument("--filter_face_detect", action='store_true', help="Enable filter: Keep only images with exactly one detected face.")
    parser.add_argument("--filter_min_face_size", type=float, default=0.20, help="Minimum face height fraction for face detection filter.")
    # --- NEW: Landmark Filter ---
    parser.add_argument("--filter_landmarks", action='store_true', help="Enable filter: Keep only images with valid basic facial landmarks.")
    parser.add_argument("--filter_sharpness", action='store_true', help="Enable filter: Keep only images above a sharpness threshold.")
    parser.add_argument("--filter_sharpness_threshold", type=float, default=50.0, help="Laplacian variance threshold for sharpness filter (Try higher values like 60-100).") # Suggest higher default

    args = parser.parse_args()

    # --- Validate Arguments ---
    # ... (keep previous validations for checkpoint, psi, num_images, batch_size) ...
    if not os.path.exists(args.checkpoint): print(f"Error: Checkpoint file not found: {args.checkpoint}"); exit(1)
    if not 0.0 <= args.truncation_psi <= 1.0: print("Error: --truncation_psi must be between 0.0 and 1.0"); exit(1)
    if args.num_images <= 0: print("Error: num_images must be positive."); exit(1)
    if args.batch_size <= 0: print("Error: batch_size must be positive."); exit(1)
    if (args.filter_face_detect or args.filter_landmarks) and not FACE_RECOGNITION_AVAILABLE:
        print("WARNING: '--filter_face_detect' or '--filter_landmarks' requested, but 'face_recognition' library is not installed or importable. These filters will be disabled.")
        args.filter_face_detect = False
        args.filter_landmarks = False
    # Ensure landmark filter requires face detect implicitly
    if args.filter_landmarks and not args.filter_face_detect:
        print("Info: Enabling --filter_face_detect as it's required by --filter_landmarks.")
        args.filter_face_detect = True
    if args.filter_face_detect and not (0.0 < args.filter_min_face_size < 1.0) : print("Error: --filter_min_face_size must be between 0 and 1"); exit(1)
    if args.filter_sharpness and args.filter_sharpness_threshold < 0: print("Error: --filter_sharpness_threshold must be non-negative"); exit(1)

    # --- Setup Device ---
    if args.device == "auto": device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = torch.device(args.device)
    print(f"Using device: {device}")
    if (args.filter_face_detect or args.filter_landmarks) and device == torch.device("cpu") and FACE_RECOGNITION_AVAILABLE:
         print("WARNING: Using face_recognition CNN model on CPU can be slow.")

        # --- Setup Seed ---
    if args.seed is not None:
        print(f"Using random seed: {args.seed}")
        seed_value = int(args.seed) # Ensure it's an integer
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        # Only set CUDA seed if CUDA is used AND seed is provided
        if device == torch.device("cuda"):
            torch.cuda.manual_seed_all(seed_value) # <-- Use seed_value here
    # No 'else' needed, if no seed is provided, don't set anything explicitly

    # --- Create Output Directory ---
    output_path = Path(args.output_dir); output_path.mkdir(parents=True, exist_ok=True); print(f"Output directory: {output_path.resolve()}")

    # --- Initialize Models ---
    netG = Generator().to(device); netD = None
    if args.filter_d: netD = Discriminator().to(device);
    if args.filter_d and not netD: print("Error: Failed to init Discriminator."); exit(1)
    print("GAN models initialized.")

    # --- Load Checkpoint ---
    print(f"Loading combined checkpoint: {args.checkpoint}")
    # ... (Loading logic remains the same) ...
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'netG_state_dict' in checkpoint: netG.load_state_dict(checkpoint['netG_state_dict'])
        else: print("Error: 'netG_state_dict' not found!"); exit(1)
        print("  Loaded Generator state.")
        if args.filter_d:
            if 'netD_state_dict' in checkpoint: netD.load_state_dict(checkpoint['netD_state_dict'])
            else: print("Error: 'netD_state_dict' not found! Cannot use --filter_d."); exit(1)
            print("  Loaded Discriminator state.")
        epoch = checkpoint.get('epoch', 'N/A'); global_step = checkpoint.get('global_step', 'N/A')
        print(f"  Checkpoint details: Epoch={epoch}, Global Step={global_step}")
    except Exception as e: print(f"Error loading checkpoint: {e}"); import traceback; traceback.print_exc(); exit(1)


    # --- Set Models to Evaluation Mode ---
    netG.eval();
    if netD: netD.eval()
    print("Models set to evaluation mode.")

    # --- Print Active Filters ---
    print("Active Generation Filters:")
    print(f"  - Truncation: {'Yes (Psi = '+str(args.truncation_psi)+')' if args.truncation_psi < 1.0 else 'No'}")
    print(f"  - Discriminator: {'Yes (Threshold > '+str(args.filter_d_threshold)+')' if args.filter_d else 'No'}")
    print(f"  - Face Detection: {'Yes (Min Height Fraction: '+str(args.filter_min_face_size)+')' if args.filter_face_detect else 'No'}")
    print(f"  - Landmarks: {'Yes' if args.filter_landmarks else 'No'}")
    print(f"  - Sharpness: {'Yes (Threshold > '+str(args.filter_sharpness_threshold)+')' if args.filter_sharpness else 'No'}")
    print("-" * 30)


    # --- Generate Images ---
    images_saved = 0; images_generated_total = 0
    images_passed_d_filter = 0; images_passed_face_filter = 0
    images_passed_landmark_filter = 0 # New counter
    images_passed_sharp_filter = 0
    target_images = args.num_images

    pbar = tqdm(total=target_images, desc="Saving Final Images")

    with torch.no_grad():
        while images_saved < target_images:
            current_batch_size = args.batch_size
            noise = torch.randn(current_batch_size, Z_DIM, 1, 1, device=device)
            images_generated_total += current_batch_size
            if args.truncation_psi < 1.0: noise = noise * args.truncation_psi
            fake_images_batch = netG(noise)

            # Apply Discriminator Filtering
            indices_passing_d = list(range(current_batch_size))
            if args.filter_d and netD:
                scores = netD(fake_images_batch)
                indices_passing_d = (scores > args.filter_d_threshold).nonzero(as_tuple=True)[0].tolist()
                images_passed_d_filter += len(indices_passing_d)

            # Apply Post-Processing Filters Sequentially
            num_saved_this_batch = 0
            for idx_in_batch in indices_passing_d:
                if images_saved >= target_images: break

                current_image_tensor = fake_images_batch[idx_in_batch]
                passed_all_filters = True # Assume passes initially

                # Convert ONLY if needed by active filters
                rgb_numpy_img = None
                if args.filter_face_detect or args.filter_landmarks or args.filter_sharpness:
                     rgb_numpy_img = tensor_to_rgb_numpy(current_image_tensor)
                     if rgb_numpy_img is None: # Handle potential conversion error
                          passed_all_filters = False; continue

                face_locations = None # Store locations if detected

                # --- Face Detection Filter ---
                if passed_all_filters and args.filter_face_detect:
                    face_locations = detect_faces_face_recognition(rgb_numpy_img)
                    if len(face_locations) != 1: passed_all_filters = False
                    else:
                        top, right, bottom, left = face_locations[0]
                        face_height = bottom - top
                        image_height = rgb_numpy_img.shape[0]
                        if (face_height / image_height) < args.filter_min_face_size:
                             passed_all_filters = False

                    if passed_all_filters: images_passed_face_filter += 1
                    else: continue # Stop processing this image

                # --- Landmark Filter ---
                # Requires face detection to have run and passed
                if passed_all_filters and args.filter_landmarks:
                    # We need face_locations from the previous step
                    if face_locations: # Ensure locations were found
                        landmark_list = get_face_landmarks(rgb_numpy_img, face_locations)
                        if not validate_landmarks(landmark_list):
                            passed_all_filters = False
                    else: # Should not happen if face_detect passed, but safety check
                        passed_all_filters = False

                    if passed_all_filters: images_passed_landmark_filter += 1
                    else: continue # Stop processing this image

                # --- Sharpness Filter ---
                if passed_all_filters and args.filter_sharpness:
                     sharpness = calculate_sharpness(rgb_numpy_img)
                     if sharpness < args.filter_sharpness_threshold:
                         passed_all_filters = False

                     if passed_all_filters: images_passed_sharp_filter += 1
                     else: continue # Stop processing this image

                # --- Save Image if All Active Filters Passed ---
                if passed_all_filters:
                    vutils.save_image(current_image_tensor, output_path / f"fake_{images_saved:06d}.png", normalize=True)
                    images_saved += 1
                    num_saved_this_batch += 1
                    pbar.update(1)

            # Update progress bar postfix
            postfix_str = f"Saved: {images_saved}/{target_images} | Gen: {images_generated_total}"
            if args.filter_d: postfix_str += f" | PassD: {images_passed_d_filter}"
            if args.filter_face_detect: postfix_str += f" | PassFace: {images_passed_face_filter}"
            if args.filter_landmarks: postfix_str += f" | PassLndmk: {images_passed_landmark_filter}" # Added landmark count
            if args.filter_sharpness: postfix_str += f" | PassSharp: {images_passed_sharp_filter}"
            pbar.set_postfix_str(postfix_str)

    pbar.close()
    print(f"\nSuccessfully saved {images_saved} images meeting all criteria in '{output_path.resolve()}'")
    print(f"  Total images generated: {images_generated_total}")
    # Print stats for each active filter
    if args.filter_d: print(f"  Images passing Discriminator filter (>{args.filter_d_threshold}): {images_passed_d_filter}")
    if args.filter_face_detect: print(f"  Images passing Face Detection filter: {images_passed_face_filter} (out of those passing prior filters)")
    if args.filter_landmarks: print(f"  Images passing Landmark filter: {images_passed_landmark_filter} (out of those passing prior filters)")
    if args.filter_sharpness: print(f"  Images passing Sharpness filter (>{args.filter_sharpness_threshold}): {images_passed_sharp_filter} (out of those passing prior filters)")