import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.color import rgb2lab
import pandas as pd
from tabulate import tabulate
import os
from tqdm import tqdm
from methods import Bilinear, HQL, HA, GBTF, RI, DLMMSE, DLMMSE1, IRI


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
    """Convert an RGB image to a Bayer pattern image."""
    if pattern not in ['RGGB', 'BGGR', 'GRBG', 'GBRG']:
        raise ValueError("Invalid Bayer pattern. Choose from 'RGGB', 'BGGR', 'GRBG', 'GBRG'.")

    new_img = np.zeros_like(img)
    mapping = {
        'RGGB': [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 2)],
        'BGGR': [(0, 0, 2), (0, 1, 1), (1, 0, 1), (1, 1, 0)],
        'GRBG': [(0, 0, 1), (0, 1, 0), (1, 0, 2), (1, 1, 1)],
        'GBRG': [(0, 0, 1), (0, 1, 2), (1, 0, 0), (1, 1, 1)],
    }

    for (i, j, c) in mapping[pattern]:
        new_img[i::2, j::2, c] = img[i::2, j::2, c]

    return new_img


# --- Metrics Calculation ---
def calculate_metrics(raw_img, new_img):
    """Calculate PSNR, SSIM, Color MSE, Edge Preservation, and Zipper Effect."""
    result = {}

    crop_raw = raw_img[5:-5, 5:-5]
    crop_new = new_img[5:-5, 5:-5]

    # PSNR and SSIM
    for c in range(3):
        result[f'PSNR_channel_{c}'] = psnr(crop_raw[:, :, c], crop_new[:, :, c])
        result[f'SSIM_channel_{c}'] = ssim(crop_raw[:, :, c], crop_new[:, :, c], data_range=255)

    result['PSNR_overall'] = psnr(crop_raw, crop_new)
    result['SSIM_overall'] = ssim(crop_raw, crop_new, data_range=255, channel_axis=2)

    # Color MSE in LAB space
    lab_raw = rgb2lab(crop_raw)
    lab_new = rgb2lab(crop_new)
    result['Color_MSE'] = np.mean(np.sum((lab_raw - lab_new) ** 2, axis=2))

    # Edge Preservation
    raw_edges = cv2.Canny(cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY), 100, 200)
    new_edges = cv2.Canny(cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY), 100, 200)
    result['Edge_preservation'] = np.sum(raw_edges & new_edges) / np.sum(raw_edges | new_edges)

    # Zipper Effect
    gray_new = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray_new, cv2.CV_64F)
    result['Zipper_effect'] = np.std(laplacian)

    return result


# --- Main Processing Loop ---
def process_images():
    """Main loop to process images and calculate metrics."""
    create_directories()
    results_data = []

    files = os.listdir("Data/input")
    if not files:
        print("No images found in 'Data/input'. Please add some images.")
        return

    for picname in files:
        print(f"Processing image: {picname}")
        file_path = os.path.join("Data/input", picname)
        src_img = load_image(file_path)

        bayer_img = make_bayer(src_img)
        
        cv2.imwrite("Data/output/GT.png", cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR))
        
        cv2.imwrite("Data/Bayer.png", cv2.cvtColor(bayer_img, cv2.COLOR_RGB2BGR))

        methods = [
            ('DLMMSE', DLMMSE.run),
        ]

        for method, func in tqdm(methods, desc="Processing algorithms"):
            print(f"\nRunning {method} algorithm...")
            demosaiced_img = func(bayer_img)
            cv2.imwrite(f"Data/output/{method}.png", cv2.cvtColor(demosaiced_img, cv2.COLOR_RGB2BGR))

            results = calculate_metrics(src_img, demosaiced_img)
            results['Method'] = method
            results_data.append(results)

            print(f"{method} algorithm completed.")

    # Create DataFrame from results
    df = pd.DataFrame(results_data)
    columns = ['Method'] + [col for col in df.columns if col != 'Method']
    df = df[columns]

    print("\nResults Table:")
    print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.4f'))

    df.to_csv('demosaicing_results.csv', index=False)
    print("\nResults exported to 'demosaicing_results.csv'")


# --- Entry Point ---
if __name__ == "__main__":
    process_images()
