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

from methods import DLMMSE,DLMMSE1

# --- Global constants ---
DATA_ROOT = "Data"
INPUT_DIR = os.path.join(DATA_ROOT, "input")
OUTPUT_DIR = os.path.join(DATA_ROOT, "output")
OUTPUT_DIR_ALGO1 = os.path.join(DATA_ROOT, "DLMMSE")
OUTPUT_DIR_ALGO2 = os.path.join(DATA_ROOT, "DLMMSE1")
# DLMMSE_DIR and DLMMSE1_DIR might not be needed if the method handles its own debug output
# DEBUG_METHOD_DIR = os.path.join(DATA_ROOT, "DLMMSE_Debug") # Example if needed

# --- Utility Functions ---
def create_directories():
    """Create necessary directories for data storage."""
    # Ensure Data, input, and general output directories exist
    dirs = [DATA_ROOT, INPUT_DIR, OUTPUT_DIR,OUTPUT_DIR_ALGO1,OUTPUT_DIR_ALGO2]
    # Add specific method debug directories if needed by the main script
    # dirs.append(DEBUG_METHOD_DIR)
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def dump_gt_channels(src_img, base_filename, output_dir=OUTPUT_DIR):
    """
    Dump the ground truth image and its individual RGB channels (grayscale).
    """
    # Save the ground truth image (BGR for OpenCV)
    gt_filename = os.path.join(output_dir, f"{base_filename}_GT.png")
    cv2.imwrite(gt_filename, cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR))

    # Save grayscale versions of each channel
    try:
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_GT_R_gray.png"), src_img[:, :, 0])
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_GT_G_gray.png"), src_img[:, :, 1])
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_GT_B_gray.png"), src_img[:, :, 2])
    except Exception as e:
        print(f"Warning: Could not save GT grayscale channels for {base_filename}: {e}")


def clear_directory(dir_path, verbose=False):
    """Clear a directory of all files and subdirectories."""
    if not os.path.exists(dir_path):
        return

    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path): # Check for links too
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
            if verbose:
                print(f"  Deleted: {item_path}")
        except Exception as e:
            if verbose:
                print(f"  Error deleting {item_path}: {e}")

def clear_output_directories(verbose=False):
    """Clear all output directories managed by this script (keeps input)."""
    if verbose:
        print("Clearing output directories...")

    # Clear general output and specific method debug dirs if defined
    dirs_to_clear = [OUTPUT_DIR]
    dirs_to_clear += [OUTPUT_DIR_ALGO1, OUTPUT_DIR_ALGO2] # Add specific method directories
    
    for dir_path in dirs_to_clear:
         if os.path.exists(dir_path):
            if verbose: print(f"Clearing: {dir_path}")
            clear_directory(dir_path, verbose)
         elif verbose:
             print(f"Directory not found, skipping clear: {dir_path}")


    if verbose:
        print("Finished clearing output directories.")

def load_image(file_path):
    """Load an image using OpenCV and convert it from BGR to RGB format."""
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Failed to load image: {file_path}")
    # Convert color space right after loading
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Bayer Pattern Conversion ---
def make_bayer(img, pattern='RGGB'):
    """Convert an RGB image to a Bayer pattern image (optimized version)."""
    pattern = pattern.upper()
    # Check for valid pattern keys from the method's dictionary if accessible,
    # or hardcode known valid patterns.
    valid_patterns = ['RGGB', 'BGGR', 'GRBG', 'GBRG']
    if pattern not in valid_patterns:
        raise ValueError(f"Invalid Bayer pattern: {pattern}. Supported: {valid_patterns}")

    # Create a zeroed image once
    new_img = np.zeros_like(img)

    # Mapping of patterns to pixel positions (row_offset, col_offset, channel_index)
    mapping = {
        'RGGB': [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 2)], # R, G1, G2, B
        'BGGR': [(0, 0, 2), (0, 1, 1), (1, 0, 1), (1, 1, 0)], # B, G1, G2, R
        'GRBG': [(0, 0, 1), (0, 1, 0), (1, 0, 2), (1, 1, 1)], # G1, R, B, G2
        'GBRG': [(0, 0, 1), (0, 1, 2), (1, 0, 0), (1, 1, 1)]  # G1, B, R, G2
    }

    # Apply the pattern using slicing for efficiency
    for row_offset, col_offset, channel_index in mapping[pattern]:
        new_img[row_offset::2, col_offset::2, channel_index] = img[row_offset::2, col_offset::2, channel_index]

    return new_img

def calculate_metrics(raw_img_input: 'numpy.ndarray',
                                    new_img_input: 'numpy.ndarray') -> dict:
    """
    Calculates various image quality metrics between a raw and a new image.

    This version encapsulates ALL dependencies (imports, constants) within
    the function itself for self-containment. Standard practice is to define
    imports and constants at the module level.

    Handles cropping, per-channel/overall PSNR/SSIM, Color MSE in LAB,
    Edge Preservation IoU, and Zipper artifact estimation.

    Args:
        raw_img_input: The original reference image (H, W, 3) as a NumPy array (uint8).
                       Assumed to be in RGB format.
        new_img_input: The processed/generated image (H, W, 3) as a NumPy array (uint8).
                       Assumed to be in RGB format.

    Returns:
        A dictionary containing the calculated metric values. Returns np.nan
        for metrics that fail to compute. Reports calculation issues via warnings.

    Raises:
        ValueError: If input images do not have the same height and width.
        ImportError: If required libraries (numpy, opencv-python, scikit-image)
                     are not installed.
    """
    # --- Encapsulated Imports (Not Standard Practice) ---
    import numpy as np
    import warnings
    try:
        import cv2
    except ImportError as e:
         raise ImportError(
             "OpenCV (cv2) is required for this function. "
             "Please install it (`pip install opencv-python`)."
         ) from e
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        from skimage.color import rgb2lab
    except ImportError as e:
        raise ImportError(
            "Scikit-image is required for this function. "
            "Please install it (`pip install scikit-image`)."
        ) from e

    # --- Encapsulated Constants (Not Standard Practice) ---
    DEFAULT_CROP_SIZE = 5
    CANNY_LOW_THRESHOLD = 100
    CANNY_HIGH_THRESHOLD = 200
    DATA_RANGE = 255  # Assumes 8-bit images (0-255)

    # --- Function Logic ---

    # Use copies to avoid modifying original inputs if they are views
    raw_img = np.copy(raw_img_input)
    new_img = np.copy(new_img_input)
    crop_size = DEFAULT_CROP_SIZE # Use the internal constant

    # --- Input Validation ---
    if raw_img.shape[:2] != new_img.shape[:2]:
        raise ValueError("Input images must have the same height and width. "
                         f"Got {raw_img.shape} and {new_img.shape}")
    if raw_img.ndim != 3 or new_img.ndim != 3 or raw_img.shape[2] != 3 or new_img.shape[2] != 3:
        # Check if grayscale was passed accidentally
        if raw_img.ndim == 2 and new_img.ndim == 2:
             warnings.warn("Input images appear to be grayscale (H, W). "
                           "Adapting for grayscale PSNR/SSIM/Edge/Zipper. Color metrics will be NaN.", UserWarning)
             # Convert grayscale to 3-channel grayscale for consistent processing where needed
             raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
             new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)
        else:
            warnings.warn(f"Expected RGB images (H, W, 3), but got shapes "
                          f"{raw_img.shape} and {new_img.shape}. Metrics might be inaccurate.", UserWarning)
            # Attempt to proceed, but results might be questionable


    # --- Initialization ---
    results = {}
    h, w = raw_img.shape[:2] # Use shape from potentially adapted images

    # --- Cropping (for PSNR, SSIM, Color MSE) ---
    # Avoids border artifacts affecting these metrics
    if h <= 2 * crop_size or w <= 2 * crop_size:
        warnings.warn(f"Image too small ({raw_img.shape[:2]}) for cropping by {crop_size} pixels. "
                      "Calculating PSNR/SSIM/ColorMSE on full image.", UserWarning)
        crop_raw = raw_img
        crop_new = new_img
    else:
        crop_raw = raw_img[crop_size:-crop_size, crop_size:-crop_size]
        crop_new = new_img[crop_size:-crop_size, crop_size:-crop_size]

    # --- Core Metrics (using cropped images) ---

    # 1. Per-Channel PSNR and SSIM
    channel_names = ['R', 'G', 'B'] if crop_raw.shape[2] == 3 else ['Gray']
    for c, name in enumerate(channel_names):
        # Handle both 3-channel and single-channel (adapted grayscale) cases
        raw_channel = crop_raw[..., c] if crop_raw.ndim == 3 else crop_raw
        new_channel = crop_new[..., c] if crop_new.ndim == 3 else crop_new

        # PSNR per channel/grayscale
        try:
            results[f'PSNR_{name}'] = psnr(raw_channel, new_channel, data_range=DATA_RANGE)
        except (ValueError, Exception) as e:
            warnings.warn(f"PSNR calculation failed for {name}: {e}", RuntimeWarning)
            results[f'PSNR_{name}'] = np.nan

        # SSIM per channel/grayscale
        try:
            results[f'SSIM_{name}'] = ssim(raw_channel, new_channel, data_range=DATA_RANGE)
        except (ValueError, Exception) as e:
            warnings.warn(f"SSIM calculation failed for {name}: {e}", RuntimeWarning)
            results[f'SSIM_{name}'] = np.nan
        # Break after one loop if grayscale
        if crop_raw.ndim == 2:
             break

    # 2. Overall PSNR (Multi-channel or Grayscale)
    try:
        results['PSNR_Overall'] = psnr(crop_raw, crop_new, data_range=DATA_RANGE)
    except (ValueError, Exception) as e:
        warnings.warn(f"Overall PSNR calculation failed: {e}", RuntimeWarning)
        results['PSNR_Overall'] = np.nan

    # 3. Overall SSIM (Multi-channel or Grayscale)
    # Default to grayscale calculation if not 3 channels
    is_multichannel = crop_raw.ndim == 3 and crop_raw.shape[2] > 1
    try:
        if is_multichannel:
            try:
                # Modern API (skimage >= 0.19): multichannel=True
                results['SSIM_Overall'] = ssim(crop_raw, crop_new, data_range=DATA_RANGE,
                                               multichannel=True, channel_axis=-1)
            except TypeError:
                 # Fallback for older API: use channel_axis
                 warnings.warn("Falling back to older SSIM API (using channel_axis).", UserWarning)
                 results['SSIM_Overall'] = ssim(crop_raw, crop_new, data_range=DATA_RANGE,
                                                multichannel=False, channel_axis=-1)
        else:
             # Grayscale SSIM calculation
             results['SSIM_Overall'] = ssim(crop_raw, crop_new, data_range=DATA_RANGE)

    except (ValueError, Exception) as e:
        warnings.warn(f"Overall SSIM calculation failed: {e}", RuntimeWarning)
        results['SSIM_Overall'] = np.nan

    # 4. Color MSE in CIELAB space (Only if input was originally color)
    if raw_img_input.ndim == 3 and raw_img_input.shape[2] == 3:
        try:
            # Use the *cropped* RGB images for LAB conversion
            lab_raw = rgb2lab(crop_raw)
            lab_new = rgb2lab(crop_new)
            delta_e_sq = np.sum(np.square(lab_raw - lab_new), axis=2)
            results['Color_MSE_LAB'] = np.mean(delta_e_sq)
        except Exception as e:
            warnings.warn(f"Color MSE (LAB) calculation failed: {e}", RuntimeWarning)
            results['Color_MSE_LAB'] = np.nan
    else:
         # Assign NaN if input wasn't color
         results['Color_MSE_LAB'] = np.nan


    # --- Additional Metrics (Optional - using full images) ---
    try:
        # Convert full original images to grayscale
        # Handle case where input was already grayscale
        gray_raw = cv2.cvtColor(raw_img_input, cv2.COLOR_RGB2GRAY) if raw_img_input.ndim == 3 else raw_img_input
        gray_new = cv2.cvtColor(new_img_input, cv2.COLOR_RGB2GRAY) if new_img_input.ndim == 3 else new_img_input

        # 5. Edge Preservation (Intersection over Union of Canny edges)
        raw_edges = cv2.Canny(gray_raw, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD) > 0 # Binary edge map
        new_edges = cv2.Canny(gray_new, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD) > 0

        intersection = np.sum(raw_edges & new_edges)
        union = np.sum(raw_edges | new_edges)
        # Avoid division by zero if no edges are detected in either image
        results['Edge_IoU'] = intersection / max(union, 1e-9) # Use smaller epsilon

        # 6. Zipper Effect Metric (Standard deviation of Laplacian on grayscale)
        # Use float64 for Laplacian calculation to avoid overflow/precision issues
        laplacian_new = cv2.Laplacian(gray_new.astype(np.float64), cv2.CV_64F)
        results['Zipper_StdLap'] = np.std(laplacian_new)

    except Exception as e:
        warnings.warn(f"Optional metrics (Edge/Zipper) failed: {e}", RuntimeWarning)
        results['Edge_IoU'] = np.nan
        results['Zipper_StdLap'] = np.nan

    return results

# --- Process a single image with a specific pattern and method ---
def process_image_pattern(task_args):
    """Wrapper function to process a single image task, suitable for parallel execution."""
    file_path, pattern, method_info, base_filename = task_args
    method_name, method_func = method_info

    try:
        # Load source image
        src_img = load_image(file_path)

        # Create Bayer pattern image
        bayer_img = make_bayer(src_img, pattern=pattern)

        # Save Bayer pattern image (optional)
        # bayer_filename = os.path.join(DATA_ROOT, f"Bayer_{pattern}_{base_filename}.png")
        # cv2.imwrite(bayer_filename, cv2.cvtColor(bayer_img, cv2.COLOR_RGB2BGR))

        # Process the image using the specified method's run function
        # Pass pattern, potentially other args if method expects them
        demosaiced_img = method_func(bayer_img, pattern=pattern) # Assuming run(img, pattern) signature

        # Save demosaiced output image (BGR for OpenCV)
        output_filename = os.path.join(OUTPUT_DIR, f"{base_filename}_{method_name}_{pattern}.png")
        cv2.imwrite(output_filename, cv2.cvtColor(demosaiced_img, cv2.COLOR_RGB2BGR))

        # Calculate metrics
        metrics = calculate_metrics(src_img, demosaiced_img)
        # Add metadata to metrics dictionary
        metrics['Method'] = method_name
        metrics['Pattern'] = pattern
        metrics['Image'] = os.path.basename(file_path)

        return metrics
    except Exception as e:
        # Log errors encountered during processing
        print(f"\nERROR processing '{os.path.basename(file_path)}' with {method_name} ({pattern}): {e}")
        # Optionally log traceback: import traceback; traceback.print_exc()
        return None # Return None indicates failure for this task

# --- Main Processing Function ---
def process_images(patterns=None, parallel=True, max_workers=None):
    """Process images with multiple Bayer patterns and demosaicing methods."""
    # Setup directories
    create_directories()
    # Clear previous results (optional, use with caution)
    clear_output_directories(verbose=True)

    # Default to RGGB pattern if none specified
    if patterns is None or not patterns:
        patterns = ['RGGB']
    print(f"Using patterns: {patterns}")

    # List input image files from INPUT_DIR
    try:
        input_files = [f for f in os.listdir(INPUT_DIR)
                       if os.path.isfile(os.path.join(INPUT_DIR, f)) and
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
    except FileNotFoundError:
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return

    if not input_files:
        print(f"No supported images found in input directory: {INPUT_DIR}")
        return
    print(f"Found {len(input_files)} images in {INPUT_DIR}")

    # --- Define the demosaicing methods to test ---
    # Each entry is a tuple: ('Method Name for Reports', method_run_function)
    methods_to_run = [
        ('DLMMSE', DLMMSE.run),
        ('DLMMSE1', DLMMSE1.run)        
    ]
    print(f"Using methods: {[name for name, func in methods_to_run]}")


    # --- Prepare ground truth and tasks ---
    tasks = []
    print("Preparing tasks and saving ground truth images...")
    for filename in tqdm(input_files, desc="Preparing GT"):
        file_path = os.path.join(INPUT_DIR, filename)
        base_filename = os.path.splitext(filename)[0]

        try:
            # Load and save ground truth image once
            src_img = load_image(file_path)
            dump_gt_channels(src_img, base_filename, OUTPUT_DIR)

            # Create tasks for this image
            for pattern in patterns:
                for method_name, method_func in methods_to_run:
                    tasks.append((file_path, pattern, (method_name, method_func), base_filename))
        except Exception as e:
            print(f"\nError preparing GT or tasks for {filename}: {e}")


    if not tasks:
        print("No processing tasks generated.")
        return

    # --- Process all tasks (Sequentially or in Parallel) ---
    results_data = []
    print(f"\nStarting processing for {len(tasks)} tasks...")

    if parallel and len(tasks) > 1:
        # Parallel processing using ProcessPoolExecutor
        if max_workers is None:
             # Default to number of CPU cores
             max_workers = os.cpu_count()
        print(f"Using parallel processing with up to {max_workers} workers.")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use executor.map for simplicity if order doesn't matter, or submit for progress bar
            futures = [executor.submit(process_image_pattern, task) for task in tasks]

            # Process results as they complete with tqdm progress bar
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures),
                               desc="Processing images"):
                try:
                    result = future.result() # Get result from future
                    if result: # Append if processing was successful
                        results_data.append(result)
                except Exception as e:
                    # This catches errors raised *within* the future's execution if not caught inside
                    print(f"\nERROR in parallel task execution: {e}")
    else:
        # Sequential processing
        print("Using sequential processing.")
        for task in tqdm(tasks, desc="Processing images"):
            result = process_image_pattern(task)
            if result: # Append if processing was successful
                results_data.append(result)

    # --- Generate and Save Results ---
    if not results_data:
        print("No results were generated (all tasks might have failed).")
        return

    print(f"\nGenerated {len(results_data)} results.")

    # Create Pandas DataFrame for easy analysis and export
    df = pd.DataFrame(results_data)

    # Define desired column order
    # Start with identifiers, then add sorted metric columns
    id_cols = ['Image', 'Pattern', 'Method']
    metric_cols = sorted([c for c in df.columns if c not in id_cols])
    col_order = id_cols + metric_cols
    # Handle cases where some columns might be missing if all metrics failed
    df = df[[col for col in col_order if col in df.columns]]


    # Save results to CSV file
    csv_filename = os.path.join(DATA_ROOT, 'demosaicing_results.csv')
    try:
        df.to_csv(csv_filename, index=False, float_format='%.6f') # Format floats in CSV
        print(f"Results exported to '{csv_filename}'")
    except Exception as e:
        print(f"Error saving results to CSV {csv_filename}: {e}")

    # --- Print Summary Tables ---
    print("\n\n--- Results Summary ---")
    # Print the full results table using tabulate
    print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.4f', showindex=False))

    # Calculate and print average metrics grouped by Method and Pattern
    try:
        # Select only numeric columns for aggregation
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols: # Proceed only if there are numeric columns
             # Group by Method and Pattern, calculate mean for numeric metrics
             df_means = df.groupby(['Method', 'Pattern'])[numeric_cols].mean().reset_index()

             print("\n\n--- Average Metrics by Method and Pattern ---")
             print(tabulate(df_means, headers='keys', tablefmt='grid', floatfmt='.4f', showindex=False))

             # --- Best method analysis (Optional) ---
             # print("\n--- Best Method Analysis (per Pattern) ---")
             # for pattern in df_means['Pattern'].unique():
             #     pattern_means = df_means[df_means['Pattern'] == pattern].set_index('Method')
             #     if pattern_means.empty: continue
             #
             #     print(f"\nPattern: {pattern}")
             #     # Find the index (Method name) of the max value for each metric column
             #     # Note: Lower is better for Color_MSE_LAB and Zipper_StdLap
             #     best_methods = {}
             #     for metric in pattern_means.columns:
             #         if 'MSE' in metric or 'Zipper' in metric: # Lower is better
             #             best_method = pattern_means[metric].idxmin()
             #             best_value = pattern_means[metric].min()
             #         else: # Higher is better for PSNR, SSIM, Edge_IoU
             #             best_method = pattern_means[metric].idxmax()
             #             best_value = pattern_means[metric].max()
             #         best_methods[metric] = (best_method, best_value)
             #
             #     for metric, (method, value) in best_methods.items():
             #          print(f"  Best {metric}: {method} ({value:.4f})")

        else:
            print("\nNo numeric metric columns found for averaging.")

    except Exception as e:
        print(f"\nError during statistical analysis: {e}")

    return df

# --- Script Entry Point ---
if __name__ == "__main__":
    # Run the image processing pipeline
    # Specify patterns to test, e.g., ['RGGB', 'GRBG'] or None for default ['RGGB']
    # Enable/disable parallel processing
    process_images(patterns=['RGGB'], parallel=True, max_workers=None) # Use None for auto CPU core detection
