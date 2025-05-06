import os
import pandas as pd
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
OUTPUT_DIR_ALGO1 = os.path.join(DATA_ROOT, "DLMMSE")
OUTPUT_DIR_ALGO2 = os.path.join(DATA_ROOT, "DLMMSE1")

def clear_output_directories(verbose=False):
    """Clear all output directories managed by this script (keeps input)."""
    if verbose:
        print("Clearing output directories...")
    dirs_to_clear = [OUTPUT_DIR, OUTPUT_DIR_ALGO1, OUTPUT_DIR_ALGO2]
    for dir_path in dirs_to_clear:
        if os.path.exists(dir_path):
            if verbose: print(f"Clearing: {dir_path}")
            clear_directory(dir_path, verbose)
        elif verbose:
            print(f"Directory not found, skipping clear: {dir_path}")
    if verbose:
        print("Finished clearing output directories.")

def process_image_pattern(task_args):
    """Wrapper function to process a single image task, suitable for parallel execution."""
    file_path, pattern, method_info, base_filename = task_args
    method_name, method_func = method_info
    try:
        src_img = load_image(file_path)
        bayer_img = make_bayer(src_img, pattern=pattern)
        demosaiced_img = method_func(bayer_img, pattern=pattern)
        output_filename = os.path.join(OUTPUT_DIR, f"{base_filename}_{method_name}_{pattern}.png")
        cv2.imwrite(output_filename, cv2.cvtColor(demosaiced_img, cv2.COLOR_RGB2BGR))
        metrics = calculate_metrics(src_img, demosaiced_img)
        metrics['Method'] = method_name
        metrics['Pattern'] = pattern
        metrics['Image'] = os.path.basename(file_path)
        return metrics
    except Exception as e:
        print(f"\nERROR processing '{os.path.basename(file_path)}' with {method_name} ({pattern}): {e}")
        return None

def process_images(patterns=None, parallel=True, max_workers=None):
    """Process images with multiple Bayer patterns and demosaicing methods."""
    create_directories([DATA_ROOT, INPUT_DIR, OUTPUT_DIR, OUTPUT_DIR_ALGO1, OUTPUT_DIR_ALGO2])
    clear_output_directories(verbose=True)
    if patterns is None or not patterns:
        patterns = ['RGGB']
    print(f"Using patterns: {patterns}")
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
    methods_to_run = [
        ('DLMMSE', DLMMSE.run),
        ('DLMMSE1', DLMMSE1.run)
    ]
    print(f"Using methods: {[name for name, func in methods_to_run]}")
    tasks = []
    print("Preparing tasks and saving ground truth images...")
    for filename in tqdm(input_files, desc="Preparing GT"):
        file_path = os.path.join(INPUT_DIR, filename)
        base_filename = os.path.splitext(filename)[0]
        try:
            src_img = load_image(file_path)
            dump_gt_channels(src_img, base_filename, OUTPUT_DIR)
            for pattern in patterns:
                for method_name, method_func in methods_to_run:
                    tasks.append((file_path, pattern, (method_name, method_func), base_filename))
        except Exception as e:
            print(f"\nError preparing GT or tasks for {filename}: {e}")
    if not tasks:
        print("No processing tasks generated.")
        return
    results_data = []
    print(f"\nStarting processing for {len(tasks)} tasks...")
    if parallel and len(tasks) > 1:
        if max_workers is None:
            max_workers = os.cpu_count()
        print(f"Using parallel processing with up to {max_workers} workers.")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_image_pattern, task) for task in tasks]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images"):
                try:
                    result = future.result()
                    if result:
                        results_data.append(result)
                except Exception as e:
                    print(f"\nERROR in parallel task execution: {e}")
    else:
        print("Using sequential processing.")
        for task in tqdm(tasks, desc="Processing images"):
            result = process_image_pattern(task)
            if result:
                results_data.append(result)
    if not results_data:
        print("No results were generated (all tasks might have failed).")
        return
    print(f"\nGenerated {len(results_data)} results.")
    df = pd.DataFrame(results_data)
    id_cols = ['Image', 'Pattern', 'Method']
    metric_cols = sorted([c for c in df.columns if c not in id_cols])
    col_order = id_cols + metric_cols
    df = df[[col for col in col_order if col in df.columns]]
    csv_filename = os.path.join(DATA_ROOT, 'demosaicing_results.csv')
    try:
        df.to_csv(csv_filename, index=False, float_format='%.6f')
        print(f"Results exported to '{csv_filename}'")
    except Exception as e:
        print(f"Error saving results to CSV {csv_filename}: {e}")
    print("\n\n--- Results Summary ---")
    print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.4f', showindex=False))
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            df_means = df.groupby(['Method', 'Pattern'])[numeric_cols].mean().reset_index()
            print("\n\n--- Average Metrics by Method and Pattern ---")
            print(tabulate(df_means, headers='keys', tablefmt='grid', floatfmt='.4f', showindex=False))
        else:
            print("\nNo numeric metric columns found for averaging.")
    except Exception as e:
        print(f"\nError during statistical analysis: {e}")
    return df

if __name__ == "__main__":
    process_images(patterns=['RGGB'], parallel=True, max_workers=None)