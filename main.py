import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.color import rgb2lab
import pandas as pd
from tabulate import tabulate
import os
import shutil
from tqdm import tqdm

# --- Import your methods here ---
from methods import Bilinear, HQL, HA, GBTF, RI, DLMMSE, DLMMSE1, IRI


def clear_output_directories(root_dir="Data"):
    """
    Deletes the contents (files and subdirectories) of all folders
    within the root directory, except for the 'input' folder.

    Args:
        root_dir (str): The root directory containing the folders.
                        Defaults to "Data".
    """
    folders_to_clear = []
    for item_name in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item_name)
        if os.path.isdir(item_path) and item_name != "input":
            folders_to_clear.append(item_path)

    print(f"Clearing contents of the following folders: {', '.join([os.path.basename(f) for f in folders_to_clear])}")

    for folder_path in folders_to_clear:
        print(f"Clearing: {folder_path}")
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"  Deleted file: {item_name}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"  Deleted directory: {item_name}")
            except Exception as e:
                print(f"  Error deleting {item_name} in {folder_path}: {e}")

    print("Finished clearing output directories.")
    
# --- Utility Functions ---
def create_directories():
    """Create necessary directories for data storage."""
    dirs = ["Data", "Data/input", "Data/output", "Data/DLMMSE", "Data/DLMMSE1"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def load_image(file_path):
    """Load and convert an image from BGR to RGB."""
    img = cv2.imread(file_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Bayer Pattern Conversion ---
def make_bayer(img, pattern='RGGB'):
    """Convert an RGB image to a Bayer pattern image.

    Args:
        img (numpy.ndarray): Input 3-channel RGB image.
        pattern (str): The Bayer pattern ('RGGB', 'BGGR', 'GRBG', 'GBRG').

    Returns:
        numpy.ndarray: 3-channel image with Bayer pattern applied (zeros at non-sampled locations).
    """
    if pattern.upper() not in ['RGGB', 'BGGR', 'GRBG', 'GBRG']:
        raise ValueError("Invalid Bayer pattern. Choose from 'RGGB', 'BGGR', 'GRBG', 'GBRG'.")

    # Use uppercase pattern for consistency
    pattern = pattern.upper()

    # Create an empty image of the same shape and type
    new_img = np.zeros_like(img)

    # Define the mapping from pattern string to (row_offset, col_offset, channel_index)
    # for the 2x2 Bayer block starting at (0,0).
    mapping = {
        'RGGB': [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 2)], # R at (0,0), G at (0,1) & (1,0), B at (1,1)
        'BGGR': [(0, 0, 2), (0, 1, 1), (1, 0, 1), (1, 1, 0)], # B at (0,0), G at (0,1) & (1,0), R at (1,1)
        'GRBG': [(0, 0, 1), (0, 1, 0), (1, 0, 2), (1, 1, 1)], # G at (0,0) & (1,1), R at (0,1), B at (1,0)
        'GBRG': [(0, 0, 1), (0, 1, 2), (1, 0, 0), (1, 1, 1)], # G at (0,0) & (1,1), B at (0,1), R at (1,0)
    }

    # Apply the Bayer pattern by copying pixels from the original image
    # to the new image only at the specified locations for each channel.
    # The slices [i::2, j::2] select pixels with row index i + 2k and col index j + 2l.
    for (i, j, c) in mapping[pattern]:
        new_img[i::2, j::2, c] = img[i::2, j::2, c]

    return new_img

# --- Metrics Calculation ---
def calculate_metrics(raw_img, new_img):
    """Calculate PSNR, SSIM, Color MSE, Edge Preservation, and Zipper Effect."""
    result = {}

    # Slightly crop images to avoid boundary artifacts
    crop_raw = raw_img[5:-5, 5:-5]
    crop_new = new_img[5:-5, 5:-5]

    # PSNR and SSIM for each channel
    for c in range(3):
        result[f'PSNR_channel_{c}'] = psnr(crop_raw[:, :, c], crop_new[:, :, c])
        result[f'SSIM_channel_{c}'] = ssim(crop_raw[:, :, c], crop_new[:, :, c], data_range=255)

    # Overall PSNR and SSIM (3-channel)
    result['PSNR_overall'] = psnr(crop_raw, crop_new)
    result['SSIM_overall'] = ssim(crop_raw, crop_new, data_range=255, channel_axis=2)

    # Color MSE in LAB space
    lab_raw = rgb2lab(crop_raw)
    lab_new = rgb2lab(crop_new)
    result['Color_MSE'] = np.mean(np.sum((lab_raw - lab_new) ** 2, axis=2))

    # Edge Preservation (using Canny edges)
    raw_edges = cv2.Canny(cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY), 100, 200)
    new_edges = cv2.Canny(cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY), 100, 200)
    # Intersection over union for edge preservation
    result['Edge_preservation'] = np.sum(raw_edges & new_edges) / np.sum(raw_edges | new_edges)

    # Zipper Effect (using Laplacian standard deviation)
    gray_new = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray_new, cv2.CV_64F)
    result['Zipper_effect'] = np.std(laplacian)

    return result

# --- Main Processing Loop ---
def process_images():
    """Main loop to process images, test multiple Bayer patterns, and calculate metrics."""
    create_directories()
    clear_output_directories()
    results_data = []
    bayer_patterns = ['RGGB', 'BGGR', 'GRBG', 'GBRG'] # Define the patterns to test

    files = os.listdir("Data/input")
    if not files:
        print("No images found in 'Data/input'. Please add some images.")
        return

    for picname in files:
        print(f"\n======================================================")
        print(f"Processing image: {picname}")
        print(f"======================================================")
        file_path = os.path.join("Data/input", picname)
        src_img = load_image(file_path)
        
        # Save the ground truth image once per input file
        gt_filename = os.path.join("Data/output", f"{os.path.splitext(picname)[0]}_GT.png")
        cv2.imwrite(gt_filename, cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR))
        print(f"Saved Ground Truth: {gt_filename}")

        # --- Loop through each Bayer pattern ---
        for pattern in bayer_patterns:
            print(f"\n--- Testing Pattern: {pattern} for image: {picname} ---")

            # Create a synthetic Bayer pattern from the ground-truth image for the current pattern
            bayer_img = make_bayer(src_img, pattern=pattern)

            # Save the specific Bayer pattern image
            bayer_filename = os.path.join("Data", f"Bayer_{pattern}_{os.path.splitext(picname)[0]}.png")
            cv2.imwrite(bayer_filename, cv2.cvtColor(bayer_img, cv2.COLOR_RGB2BGR))
            print(f"Saved Bayer Image ({pattern}): {bayer_filename}")

            # List the DLMMSE method to run
            # Each entry is a tuple of (method_name, function, pattern)
            methods = [
                ('DLMMSE', DLMMSE.run, pattern)                
            ]

            # Store per-pattern metrics in a dict keyed by method for comparison
            per_pattern_results = {}

            for method_name, func, current_pattern in tqdm(methods, desc=f"Processing algorithms for {pattern}"):
                print(f"\nRunning {method_name} algorithm on {current_pattern} pattern...")
                try:
                    # Pass the pattern parameter to the function
                    demosaiced_img = func(bayer_img, pattern=current_pattern)
                except Exception as e:
                     print(f"ERROR running {method_name} on {current_pattern} for {picname}: {e}")
                     print("Skipping metrics calculation for this method/pattern.")
                     continue # Skip to the next method if execution fails


                output_filename = os.path.join("Data/output", f"{os.path.splitext(picname)[0]}_{method_name}_{current_pattern}.png")
                cv2.imwrite(output_filename, cv2.cvtColor(demosaiced_img, cv2.COLOR_RGB2BGR))
                print(f"Saved Demosaiced Image: {output_filename}")

                # Calculate metrics
                metrics = calculate_metrics(src_img, demosaiced_img)
                metrics['Method'] = method_name
                metrics['Pattern'] = current_pattern # Add the pattern info
                metrics['Image'] = picname  # keep track of which image was processed
                results_data.append(metrics)
                per_pattern_results[method_name] = metrics

                print(f"{method_name} algorithm ({current_pattern}) completed.")

            # --- Compare the two methods on this image *for this specific pattern* ---
            if 'DLMMSE' in per_pattern_results and 'DLMMSE1' in per_pattern_results:
                dlmmse_metrics = per_pattern_results['DLMMSE']
                dlmmse1_metrics = per_pattern_results['DLMMSE1']

                # Print which method is better for this image and pattern
                better_psnr = 'DLMMSE' if dlmmse_metrics['PSNR_overall'] >= dlmmse1_metrics['PSNR_overall'] else 'DLMMSE1'
                better_ssim = 'DLMMSE' if dlmmse_metrics['SSIM_overall'] >= dlmmse1_metrics['SSIM_overall'] else 'DLMMSE1'

                print(f"\n[Comparison for {picname} ({pattern})]:")
                print(f"  - PSNR_overall: DLMMSE = {dlmmse_metrics['PSNR_overall']:.4f}, DLMMSE1 = {dlmmse1_metrics['PSNR_overall']:.4f}")
                print(f"    -> Better PSNR ({pattern}): {better_psnr}")
                print(f"  - SSIM_overall: DLMMSE = {dlmmse_metrics['SSIM_overall']:.4f}, DLMMSE1 = {dlmmse1_metrics['SSIM_overall']:.4f}")
                print(f"    -> Better SSIM ({pattern}): {better_ssim}")
            else:
                print(f"\n[Comparison for {picname} ({pattern})]: Could not compare methods (one or both failed/skipped).")


    # --- Aggregate Results and Reporting ---
    if not results_data:
        print("\nNo results were generated. Exiting.")
        return

    # Create a DataFrame from the aggregated results
    df = pd.DataFrame(results_data)

    # Reorder columns to put 'Image', 'Pattern', and 'Method' first
    col_order = ['Image', 'Pattern', 'Method'] + sorted([c for c in df.columns if c not in ['Image', 'Pattern', 'Method']])
    df = df[col_order]

    # Print the results table for all processed images and patterns
    print("\n\n======================================================")
    print("           Final Results Table (All Images & Patterns)")
    print("======================================================")
    # Use maxcolwidths to prevent excessive wrapping in the console if needed
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
         print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.4f', showindex=False))


    # Save to CSV
    csv_filename = 'demosaicing_results_all_patterns.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\nResults exported to '{csv_filename}'")

    # --- Overall Comparison: Which method is better on average per pattern? ---
    # Group by 'Method' and 'Pattern' and compute the mean of numeric metrics
    try:
        # Select only numeric columns before grouping and calculating mean
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        group_cols = ['Method', 'Pattern']
        if not all(item in df.columns for item in group_cols):
             print("\nCould not perform average metric calculation: Missing 'Method' or 'Pattern' columns.")
        else:
            df_means = df.groupby(group_cols)[numeric_cols].mean()

            print("\n\n======================================================")
            print("        Average Metrics across all images (Grouped by Method & Pattern)")
            print("======================================================")
            print(tabulate(df_means, headers='keys', tablefmt='grid', floatfmt='.4f'))

            # Identify best method per metric *for each pattern*
            print("\n--- Best Method Analysis (per Pattern) ---")
            for pattern in bayer_patterns:
                print(f"\nAnalysis for Pattern: {pattern}")
                pattern_means = df_means.loc[(slice(None), pattern), :] # Select rows for the current pattern
                if pattern_means.empty:
                    print("  No data available for this pattern.")
                    continue
                # Find the index (method name) of the max value for each metric column
                best_methods_for_pattern = pattern_means.idxmax()
                for metric in pattern_means.columns:
                     # idxmax() returns the index (Method, Pattern) tuple, we only need the Method part
                     best_method_tuple = best_methods_for_pattern[metric]
                     # Ensure it's a tuple before accessing index 0
                     best_method_name = best_method_tuple[0] if isinstance(best_method_tuple, tuple) else best_method_tuple
                     best_value = pattern_means.loc[best_method_tuple, metric]
                     print(f"  - Best method for {metric}: {best_method_name} (avg value = {best_value:.4f})")

    except Exception as e:
        print(f"\nError during average metric calculation: {e}")

# --- Entry Point ---
if __name__ == "__main__":
    process_images()
