import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.color import rgb2lab
import pandas as pd
from tabulate import tabulate
import os
import shutil
from tqdm import tqdm
import concurrent.futures
from functools import partial

# --- Import your methods here ---
from methods import DLMMSE  # Import only the methods you're using

# --- Global constants ---
DATA_ROOT = "Data"
INPUT_DIR = os.path.join(DATA_ROOT, "input")
OUTPUT_DIR = os.path.join(DATA_ROOT, "output")
DLMMSE_DIR = os.path.join(DATA_ROOT, "DLMMSE")

# --- Utility Functions ---
def create_directories():
    """Create necessary directories for data storage."""
    dirs = [DATA_ROOT, INPUT_DIR, OUTPUT_DIR, DLMMSE_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
def dump_gt_channels(src_img, base_filename, output_dir=OUTPUT_DIR):
    """
    Dump the ground truth image and its individual RGB channels.
    
    Args:
        src_img (numpy.ndarray): Source RGB image
        base_filename (str): Base name for the output files
        output_dir (str): Directory to save the images
    """
    # Save the ground truth image
    gt_filename = os.path.join(output_dir, f"{base_filename}_GT.png")
    cv2.imwrite(gt_filename, cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR))
    
    # Extract and save individual channels
    height, width, _ = src_img.shape
    
    # Create single channel images
    r_channel = np.zeros_like(src_img)
    g_channel = np.zeros_like(src_img)
    b_channel = np.zeros_like(src_img)
    
    # Copy each channel to the corresponding image
    r_channel[:, :, 0] = src_img[:, :, 0]  # Copy R to red channel
    g_channel[:, :, 1] = src_img[:, :, 1]  # Copy G to green channel
    b_channel[:, :, 2] = src_img[:, :, 2]  # Copy B to blue channel
    
    # Save grayscale versions of each channel
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_GT_R_gray.png"), 
                src_img[:, :, 0])
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_GT_G_gray.png"), 
                src_img[:, :, 1])
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_GT_B_gray.png"), 
                src_img[:, :, 2])

def clear_directory(dir_path, verbose=False):
    """Clear a directory of all files and subdirectories."""
    if not os.path.exists(dir_path):
        return
    
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        try:
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
            if verbose:
                print(f"  Deleted: {item_path}")
        except Exception as e:
            if verbose:
                print(f"  Error deleting {item_path}: {e}")

def clear_output_directories(verbose=False):
    """Clear all output directories except input."""
    if verbose:
        print("Clearing output directories...")
    
    # Get all directories in DATA_ROOT
    for item in os.listdir(DATA_ROOT):
        item_path = os.path.join(DATA_ROOT, item)
        if os.path.isdir(item_path) and item != "input":
            if verbose:
                print(f"Clearing: {item_path}")
            clear_directory(item_path, verbose)
    
    if verbose:
        print("Finished clearing output directories.")

def load_image(file_path):
    """Load an image and convert it from BGR to RGB format."""
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Failed to load image: {file_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Bayer Pattern Conversion ---
def make_bayer(img, pattern='RGGB'):
    """Convert an RGB image to a Bayer pattern image (optimized version)."""
    pattern = pattern.upper()
    if pattern not in ['RGGB', 'BGGR', 'GRBG', 'GBRG']:
        raise ValueError(f"Invalid Bayer pattern: {pattern}")

    # Create a zeroed image once
    new_img = np.zeros_like(img)
    
    # Mapping of patterns to pixel positions
    mapping = {
        'RGGB': [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 2)],
        'BGGR': [(0, 0, 2), (0, 1, 1), (1, 0, 1), (1, 1, 0)],
        'GRBG': [(0, 0, 1), (0, 1, 0), (1, 0, 2), (1, 1, 1)],
        'GBRG': [(0, 0, 1), (0, 1, 2), (1, 0, 0), (1, 1, 1)]
    }
    
    # Apply the pattern (vectorized approach)
    for i, j, c in mapping[pattern]:
        new_img[i::2, j::2, c] = img[i::2, j::2, c]
        
    return new_img

# --- Metrics Calculation ---
def calculate_metrics(raw_img, new_img):
    """Calculate image quality metrics (optimized version)."""
    # Pre-calculate cropped images - avoid repetitive slicing
    crop_raw = raw_img[5:-5, 5:-5]
    crop_new = new_img[5:-5, 5:-5]
    
    # Initialize results dictionary with expected keys
    result = {}
    
    # Calculate PSNR and SSIM for individual channels
    for c in range(3):
        # Extract single channels to avoid redundant indexing
        raw_channel = crop_raw[:, :, c]
        new_channel = crop_new[:, :, c]
        
        result[f'PSNR_channel_{c}'] = psnr(raw_channel, new_channel)
        result[f'SSIM_channel_{c}'] = ssim(raw_channel, new_channel, data_range=255)
    
    # Overall metrics
    result['PSNR_overall'] = psnr(crop_raw, crop_new)
    result['SSIM_overall'] = ssim(crop_raw, crop_new, data_range=255, channel_axis=2)
    
    # Color MSE in LAB space - convert once and reuse
    lab_raw = rgb2lab(crop_raw)
    lab_new = rgb2lab(crop_new)
    result['Color_MSE'] = np.mean(np.sum(np.square(lab_raw - lab_new), axis=2))
    
    # Convert to grayscale once for edge detection
    gray_raw = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
    gray_new = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
    
    # Edge preservation
    raw_edges = cv2.Canny(gray_raw, 100, 200)
    new_edges = cv2.Canny(gray_new, 100, 200)
    # Avoid division by zero with np.maximum
    intersection = np.sum(raw_edges & new_edges)
    union = np.sum(raw_edges | new_edges)
    result['Edge_preservation'] = intersection / max(union, 1)  # Avoid division by zero
    
    # Zipper effect
    laplacian = cv2.Laplacian(gray_new, cv2.CV_64F)
    result['Zipper_effect'] = np.std(laplacian)
    
    return result

# --- Process a single image with a specific pattern and method ---
def process_image_pattern(file_path, pattern, method_info, base_filename):
    """Process a single image with a specific pattern and method."""
    method_name, func = method_info
    
    # Load source image
    src_img = load_image(file_path)
    
    dump_gt_channels(src_img, base_filename, OUTPUT_DIR)
    
    # Create Bayer pattern
    bayer_img = make_bayer(src_img, pattern=pattern)
    
    # Save Bayer pattern image if needed
    bayer_filename = os.path.join(DATA_ROOT, f"Bayer_{pattern}_{base_filename}.png")
    cv2.imwrite(bayer_filename, cv2.cvtColor(bayer_img, cv2.COLOR_RGB2BGR))
    
    try:
        # Process the image
        demosaiced_img = func(bayer_img, pattern=pattern)
        
        # Save output
        output_filename = os.path.join(OUTPUT_DIR, f"{base_filename}_{method_name}_{pattern}.png")
        cv2.imwrite(output_filename, cv2.cvtColor(demosaiced_img, cv2.COLOR_RGB2BGR))
        
        # Calculate metrics
        metrics = calculate_metrics(src_img, demosaiced_img)
        metrics['Method'] = method_name
        metrics['Pattern'] = pattern
        metrics['Image'] = os.path.basename(file_path)
        
        return metrics
    except Exception as e:
        print(f"ERROR processing {os.path.basename(file_path)} with {method_name} on {pattern}: {e}")
        return None

# --- Main Processing Function ---
def process_images(patterns=None, parallel=True, max_workers=None):
    """Process images with multiple Bayer patterns and demosaicing methods."""
    # Setup
    create_directories()
    clear_output_directories()
    
    # Use all patterns if none specified
    if patterns is None:
        patterns = ['RGGB']
    
    # List input files
    input_files = [f for f in os.listdir(INPUT_DIR) 
                  if os.path.isfile(os.path.join(INPUT_DIR, f)) and 
                  f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    if not input_files:
        print("No images found in input directory.")
        return
    
    # Set up the demosaicing methods to test
    methods = [('DLMMSE', DLMMSE.run)]
    
    # Process ground truth images first
    for filename in input_files:
        file_path = os.path.join(INPUT_DIR, filename)
        base_filename = os.path.splitext(filename)[0]
        
        # Load and save ground truth image
        src_img = load_image(file_path)
        gt_filename = os.path.join(OUTPUT_DIR, f"{base_filename}_GT.png")
        cv2.imwrite(gt_filename, cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR))
    
    # Generate all processing tasks
    tasks = []
    for filename in input_files:
        file_path = os.path.join(INPUT_DIR, filename)
        base_filename = os.path.splitext(filename)[0]
        
        for pattern in patterns:
            for method in methods:
                tasks.append((file_path, pattern, method, base_filename))
    
    # Process all tasks
    results_data = []
    
    if parallel and len(tasks) > 1:
        # Parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_image_pattern, *task) for task in tasks]
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=len(futures), 
                              desc="Processing images"):
                result = future.result()
                if result:
                    results_data.append(result)
    else:
        # Sequential processing
        for task in tqdm(tasks, desc="Processing images"):
            result = process_image_pattern(*task)
            if result:
                results_data.append(result)
    
    # Generate and save results
    if not results_data:
        print("No results were generated.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results_data)
    
    # Reorder columns
    col_order = ['Image', 'Pattern', 'Method'] + sorted([c for c in df.columns 
                                                       if c not in ['Image', 'Pattern', 'Method']])
    df = df[col_order]
    
    # Save to CSV
    csv_filename = os.path.join(DATA_ROOT, 'demosaicing_results_all_patterns.csv')
    df.to_csv(csv_filename, index=False)
    print(f"\nResults exported to '{csv_filename}'")
    
    # Print summary table
    print("\n\nFinal Results Summary:")
    print("=" * 50)
    
    # Print more compact table
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.4f', showindex=False))
    
    # Calculate aggregate statistics
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        df_means = df.groupby(['Method', 'Pattern'])[numeric_cols].mean()
        
        print("\n\nAverage Metrics by Method and Pattern:")
        print("=" * 50)
        print(tabulate(df_means, headers='keys', tablefmt='grid', floatfmt='.4f'))
        
        # Best method analysis
        print("\nBest Method Analysis (per Pattern):")
        for pattern in patterns:
            pattern_means = df_means.loc[(slice(None), pattern), :]
            if pattern_means.empty:
                continue
                
            print(f"\nPattern: {pattern}")
            best_methods = pattern_means.idxmax()
            
            for metric in pattern_means.columns:
                best_method_tuple = best_methods[metric]
                best_method = best_method_tuple[0] if isinstance(best_method_tuple, tuple) else best_method_tuple
                best_value = pattern_means.loc[best_method_tuple, metric]
                print(f"  {metric}: {best_method} ({best_value:.4f})")
                
    except Exception as e:
        print(f"\nError during statistical analysis: {e}")
    
    return df

# --- Entry Point ---
if __name__ == "__main__":
    process_images(parallel=True)