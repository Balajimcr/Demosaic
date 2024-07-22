import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.color import rgb2lab
import pandas as pd
from tabulate import tabulate
import os
from tabulate import tabulate
from tqdm import tqdm

from methods import Bilinear, HQL, HA, GBTF, RI, DLMMSE, IRI

def make_bayer(img):
    new_img = np.zeros_like(img)

    new_img[0::2, 0::2, 0] = img[0::2, 0::2, 0]
    new_img[0::2, 1::2, 1] = img[0::2, 1::2, 1]
    new_img[1::2, 0::2, 1] = img[1::2, 0::2, 1]
    new_img[1::2, 1::2, 2] = img[1::2, 1::2, 2]

    return new_img

def metrics(raw_img, new_img):
    result = {}
    
    # PSNR for each channel and overall
    for c in range(3):
        result[f'PSNR_channel_{c}'] = psnr(raw_img[5:-5, 5:-5, c], new_img[5:-5, 5:-5, c])
    result['PSNR_overall'] = psnr(raw_img[5:-5, 5:-5], new_img[5:-5, 5:-5])
    
    # SSIM for each channel and overall
    for c in range(3):
        result[f'SSIM_channel_{c}'] = ssim(raw_img[5:-5, 5:-5, c], new_img[5:-5, 5:-5, c], data_range=255)
    
    # Calculate SSIM for overall image
    result['SSIM_overall'] = ssim(raw_img[5:-5, 5:-5], new_img[5:-5, 5:-5], data_range=255, channel_axis=2)
    
    # Color accuracy in LAB color space
    lab_raw = rgb2lab(raw_img[5:-5, 5:-5])
    lab_new = rgb2lab(new_img[5:-5, 5:-5])
    result['Color_MSE'] = np.mean(np.sum((lab_raw - lab_new)**2, axis=2))
    
    # Edge preservation
    raw_edges = cv2.Canny(cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY), 100, 200)
    new_edges = cv2.Canny(cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY), 100, 200)
    result['Edge_preservation'] = np.sum(raw_edges & new_edges) / np.sum(raw_edges | new_edges)
    
    # Zipper effect detection
    gray_new = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray_new, cv2.CV_64F)
    result['Zipper_effect'] = np.std(laplacian)
    
    return result

# Create a directory to store output images
os.makedirs("Data", exist_ok=True)

results_data = []

for picname in ['./kodim19.png']:
    print(f"Processing image: {picname}")
    src_img = cv2.imread(picname)
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

    bayer_img = make_bayer(src_img)
    cv2.imwrite("Data/Bayer.png", cv2.cvtColor(bayer_img, cv2.COLOR_RGB2BGR))

    methods = [
        ('Bilinear', Bilinear.run),
        ('HQL', HQL.run),
        ('HA', HA.run),
        ('DLMMSE', DLMMSE.run),
        ('GBTF', GBTF.run),
        ('RI', RI.run),
        ('IRI', IRI.run)
    ]

    for method, func in tqdm(methods, desc="Processing algorithms"):
        print(f"\nRunning {method} algorithm...")
        demosaiced_img = func(bayer_img)
        cv2.imwrite(f"Data/{method}.png", cv2.cvtColor(demosaiced_img, cv2.COLOR_RGB2BGR))
        
        results = metrics(src_img, demosaiced_img)
        results['Method'] = method
        results_data.append(results)
        print(f"{method} algorithm completed.")

# Create a DataFrame from the results
df = pd.DataFrame(results_data)

# Reorder columns to have 'Method' first
columns = ['Method'] + [col for col in df.columns if col != 'Method']
df = df[columns]

# Print the table
print("\nResults Table:")
print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.4f'))

# Export to CSV
df.to_csv('demosaicing_results.csv', index=False)

print("\nResults have been exported to 'demosaicing_results.csv'")
print("Output images have been saved in the 'Data' directory")