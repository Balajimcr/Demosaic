import os
import pandas as pd
import numpy as np
import cv2
from tabulate import tabulate
from tqdm import tqdm
import concurrent.futures
from functools import partial

from methods.Utils import create_directories, dump_gt_channels, clear_directory, load_image, make_bayer, calculate_metrics
from methods import DLMMSE, DLMMSE1


# --- Global constants ---
DATA_ROOT = "Data"
INPUT_DIR = os.path.join(DATA_ROOT, "input")
OUTPUT_DIR = os.path.join(DATA_ROOT, "output")
# New directory for difference images
DIFFERENCE_DIR = os.path.join(OUTPUT_DIR, "differences")
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

# --- Define the desired order of metric columns ---
# Adjusted based on the headers in your shared text output
DESIRED_METRIC_ORDER = [
    'PSNR_R', 'PSNR_G', 'PSNR_B', 'PSNR_Overall',
    'SSIM_R', 'SSIM_G', 'SSIM_B', 'SSIM_Overall',
    'Color_MSE_LAB',
    'CNR_Cb_Var', 'CNR_Cr_Var', # Added CNR metrics
    'Edge_IoU', 'Zipper_StdLap'
]

def clear_output_directories(verbose: bool = False):
    """Clear all output directories managed by this script (keeps input)."""
    if verbose:
        print("Clearing output directories...")
    # List all output directories, excluding the input directory
    dirs_to_clear = [
        OUTPUT_DIR,           # Final demosaiced outputs
        DIFFERENCE_DIR,      # Difference images or comparison metrics
        os.path.join("Data", "DLMMSE"),
        os.path.join("Data", "DLMMSE1"),
    ]
    for dir_path in dirs_to_clear:
        if os.path.exists(dir_path):
            if verbose:
                print(f"Clearing: {dir_path}")
            clear_directory(dir_path, verbose)
        elif verbose:
            print(f"Directory not found, skipping clear: {dir_path}")
    if verbose:
        print("Finished clearing output directories.")

def save_difference_images(gt_img: np.ndarray, demosaiced_img: np.ndarray, base_filename: str, method_name: str, pattern: str, output_diff_dir: str):
    """
    Calculates and saves difference images between GT and demosaiced images.

    Args:
        gt_img (np.ndarray): Ground truth image (RGB format).
        demosaiced_img (np.ndarray): Demosaiced image (RGB format).
        base_filename (str): Base name of the original image file.
        method_name (str): Name of the demosaicing method used.
        pattern (str): Bayer pattern used.
        output_diff_dir (str): Root directory to save difference images into.
    """
    # Create a specific directory for this task's differences
    task_diff_dir = os.path.join(output_diff_dir, f"{base_filename}_{method_name}_{pattern}")
    os.makedirs(task_diff_dir, exist_ok=True)

    try:
        # Ensure images are in a format suitable for cv2 operations (e.g., BGR)
        # Assuming load_image and demosaicing methods output RGB, convert to BGR
        gt_bgr = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
        demosaiced_bgr = cv2.cvtColor(demosaiced_img, cv2.COLOR_RGB2BGR)

        # Calculate absolute differences per channel
        # cv2.absdiff works element-wise and handles different image depths (e.g., 8-bit, 16-bit)
        diff_b = cv2.absdiff(gt_bgr[:,:,0], demosaiced_bgr[:,:,0])
        diff_g = cv2.absdiff(gt_bgr[:,:,1], demosaiced_bgr[:,:,1])
        diff_r = cv2.absdiff(gt_bgr[:,:,2], demosaiced_bgr[:,:,2])

        # Calculate overall difference magnitude (e.g., max difference across channels)
        # np.max(..., axis=2) computes the max difference for each pixel across the 3 channels
        abs_diff_bgr = cv2.absdiff(gt_bgr, demosaiced_bgr)
        overall_diff_magnitude = np.max(abs_diff_bgr, axis=2)

        # Convert difference images to 8-bit grayscale for better visualization and saving
        # cv2.convertScaleAbs scales the absolute value to 0-255 and converts to uint8
        diff_r_8bit = cv2.convertScaleAbs(diff_r)
        diff_g_8bit = cv2.convertScaleAbs(diff_g)
        diff_b_8bit = cv2.convertScaleAbs(diff_b)
        overall_diff_8bit = cv2.convertScaleAbs(overall_diff_magnitude)


        # Save the difference images
        cv2.imwrite(os.path.join(task_diff_dir, "R_channel_diff.png"), diff_r_8bit)
        cv2.imwrite(os.path.join(task_diff_dir, "G_channel_diff.png"), diff_g_8bit)
        cv2.imwrite(os.path.join(task_diff_dir, "B_channel_diff.png"), diff_b_8bit)
        cv2.imwrite(os.path.join(task_diff_dir, "Overall_diff.png"), overall_diff_8bit)

    except Exception as e:
        print(f"\nERROR saving difference images for {base_filename}_{method_name}_{pattern}: {type(e).__name__} - {e}")


def process_image_pattern(task_args):
    """Wrapper function to process a single image task, suitable for parallel execution."""
    file_path, pattern, method_info, base_filename = task_args
    method_name, method_func = method_info

    # Optional: detailed log for each task start (can be verbose with many tasks)
    # print(f"Processing: {os.path.basename(file_path)} with {method_name} ({pattern})")

    try:
        src_img = load_image(file_path) # Assuming load_image returns RGB
        if src_img is None:
             return None

        bayer_img = make_bayer(src_img, pattern=pattern)
        if bayer_img is None:
            print(f"\nCould not create Bayer image for {os.path.basename(file_path)} with pattern {pattern}")
            return None

        demosaiced_img = method_func(bayer_img, pattern=pattern) # Assuming method_func returns RGB
        if demosaiced_img is None:
             print(f"\nDemosaicing failed for {os.path.basename(file_path)} with {method_name} ({pattern})")
             return None

        # --- Add saving of difference images ---
        save_difference_images(src_img, demosaiced_img, base_filename, method_name, pattern, DIFFERENCE_DIR)
        # --------------------------------------

        # Save the demosaiced image
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_filename = os.path.join(OUTPUT_DIR, f"{base_filename}_{method_name}_{pattern}.png")
        # Convert demosaiced_img (assuming RGB) to BGR for cv2.imwrite
        cv2.imwrite(output_filename, cv2.cvtColor(demosaiced_img, cv2.COLOR_RGB2BGR))

        # Calculate and return metrics
        metrics = calculate_metrics(src_img, demosaiced_img)
        if metrics is None:
             print(f"\nMetric calculation failed for {os.path.basename(file_path)} with {method_name} ({pattern})")
             return None

        metrics['Method'] = method_name
        metrics['Pattern'] = pattern
        metrics['Image'] = os.path.basename(file_path)

        # Optional: detailed log for each task end
        # print(f"Finished: {os.path.basename(file_path)} with {method_name} ({pattern})")
        return metrics

    except Exception as e:
        print(f"\nERROR processing '{os.path.basename(file_path)}' with {method_name} ({pattern}): {type(e).__name__} - {e}")
        return None


def process_images(patterns: list = None, parallel: bool = True, max_workers: int = None):
    """
    Process images with multiple Bayer patterns and demosaicing methods and report metrics.

    Args:
        patterns (list, optional): List of Bayer patterns to use (e.g., ['RGGB', 'BGGR']).
                                   Defaults to ['RGGB'].
        parallel (bool, optional): Whether to use parallel processing. Defaults to True.
        max_workers (int, optional): Maximum number of worker processes for parallel execution.
                                     Defaults to number of CPU cores if None and parallel is True.

    Returns:
        pd.DataFrame: DataFrame containing the results and metrics, or None if no results.
    """
    # Added DIFFERENCE_DIR to directories to create
    create_directories([DATA_ROOT, INPUT_DIR, OUTPUT_DIR, DIFFERENCE_DIR])
    clear_output_directories(verbose=True) # This will now clear the difference directory too

    if patterns is None or not patterns:
        patterns = ['RGGB']
    print(f"Using patterns: {patterns}")

    try:
        input_files = [f for f in os.listdir(INPUT_DIR)
                       if os.path.isfile(os.path.join(INPUT_DIR, f)) and
                       f.lower().endswith(IMAGE_EXTENSIONS)]
    except FileNotFoundError:
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return None

    if not input_files:
        print(f"No supported images found in input directory: {INPUT_DIR}")
        return None

    print(f"Found {len(input_files)} images in {INPUT_DIR}")

    methods_to_run = [
        ('DLMMSE', DLMMSE.run),
        ('DLMMSE1', DLMMSE1.run)
        # Add other methods here as tuples ('MethodName', method_function)
    ]
    print(f"Using methods: {[name for name, func in methods_to_run]}")

    tasks = []
    print("Preparing tasks and saving ground truth images...")
    for filename in tqdm(input_files, desc="Preparing GT and Tasks"):
        file_path = os.path.join(INPUT_DIR, filename)
        base_filename = os.path.splitext(filename)[0]
        try:
            src_img = load_image(file_path) # Assuming load_image returns RGB
            if src_img is not None:
                # Save ground truth channels once per image
                dump_gt_channels(src_img, base_filename, OUTPUT_DIR)
                # Create tasks for each pattern and method
                for pattern in patterns:
                    for method_name, method_func in methods_to_run:
                        tasks.append((file_path, pattern, (method_name, method_func), base_filename))
            else:
                 print(f"\nSkipping task generation for {filename} due to loading error.")

        except Exception as e:
            print(f"\nError preparing GT or tasks for {filename}: {type(e).__name__} - {e}")


    if not tasks:
        print("No processing tasks generated.")
        return None

    results_data = []
    print(f"\nStarting processing for {len(tasks)} tasks...")

    if parallel and len(tasks) > 1:
        if max_workers is None:
            max_workers = os.cpu_count()
            if max_workers is None or max_workers < 1:
                 max_workers = 1 # Ensure at least 1 worker if detection fails
            print(f"Using parallel processing with up to {max_workers} workers.")

        # Use ProcessPoolExecutor as a context manager
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_image_pattern, task): task for task in tasks}
            # Wrap the futures iterator with tqdm for a progress bar
            progress_bar = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images")

            for future in progress_bar:
                 try:
                     # Get result or raise exception from worker (handled within process_image_pattern)
                     result = future.result()
                     if result:
                         results_data.append(result)
                 except Exception as e:
                     # This catches exceptions during submission or result retrieval
                     print(f"\nERROR during parallel task execution or result retrieval: {type(e).__name__} - {e}")

        progress_bar.close() # Ensure progress bar is closed after the loop

    else:
        print("Using sequential processing.")
        # Sequential execution
        for task in tqdm(tasks, desc="Processing images"):
            result = process_image_pattern(task)
            if result:
                results_data.append(result)


    if not results_data:
        print("No results were generated (all tasks might have failed).")
        return None

    print(f"\nGenerated {len(results_data)} results.")

    # --- Results Reporting ---
    try:
        df = pd.DataFrame(results_data)

        # Define the full column order
        id_cols = ['Image', 'Pattern', 'Method']

        # Combine ID columns and desired metric columns, ensuring they exist in the DataFrame
        col_order = [col for col in id_cols + DESIRED_METRIC_ORDER if col in df.columns]

        # Reindex the DataFrame to apply the new column order for saving and subsequent operations
        df = df.reindex(columns=col_order)

        csv_filename = os.path.join(DATA_ROOT, 'demosaicing_results.csv')
        df.to_csv(csv_filename, index=False, float_format='%.6f')
        print(f"Results exported to '{csv_filename}'")

        print("\n\n--- Results Summary ---")
        # Pass the DataFrame AND the desired column order explicitly to tabulate
        print(tabulate(df, headers=col_order, tablefmt='grid', floatfmt='.4f', showindex=False))

        # Calculate and display average metrics
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:
            df_means = df.groupby(['Method', 'Pattern'])[numeric_cols].mean().reset_index()

            # Define the column order specifically for the average table
            avg_metric_cols_order = [col for col in DESIRED_METRIC_ORDER if col in numeric_cols]
            avg_col_order = [col for col in id_cols + avg_metric_cols_order if col in df_means.columns]

            # Reindex the means DataFrame
            df_means = df_means.reindex(columns=avg_col_order)


            print("\n\n--- Average Metrics by Method and Pattern ---")
            print(tabulate(df_means, headers=avg_col_order, tablefmt='grid', floatfmt='.4f', showindex=False))
        else:
             print("\nNo numeric metric columns found for averaging.")


    except Exception as e:
        print(f"\nError during results processing or reporting: {type(e).__name__} - {e}")
        return df

    return df

if __name__ == "__main__":
    print("Starting image processing script...")
    # Example call:
    results_df = process_images(patterns=['RGGB'], parallel=True, max_workers=None)

    if results_df is not None:
        print("\nScript finished successfully. Results DataFrame returned.")
    else:
        print("\nScript finished, but no results DataFrame was generated.")