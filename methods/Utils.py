import cv2
import numpy as np
import os
import shutil
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.color import rgb2lab
import warnings

def create_directories(dirs):
    """Create necessary directories for data storage."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def dump_gt_channels(src_img, base_filename, output_dir):
    """
    Dump the ground truth image and its individual RGB channels (grayscale).
    """
    gt_filename = os.path.join(output_dir, f"{base_filename}_GT.png")
    cv2.imwrite(gt_filename, cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR))
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
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
            if verbose:
                print(f"  Deleted: {item_path}")
        except Exception as e:
            if verbose:
                print(f"  Error deleting {item_path}: {e}")

def load_image(file_path):
    """Load an image using OpenCV and convert it from BGR to RGB format."""
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Failed to load image: {file_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def make_bayer(img, pattern='RGGB'):
    """Convert an RGB image to a Bayer pattern image (optimized version)."""
    pattern = pattern.upper()
    valid_patterns = ['RGGB', 'BGGR', 'GRBG', 'GBRG']
    if pattern not in valid_patterns:
        raise ValueError(f"Invalid Bayer pattern: {pattern}. Supported: {valid_patterns}")
    new_img = np.zeros_like(img)
    mapping = {
        'RGGB': [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 2)],
        'BGGR': [(0, 0, 2), (0, 1, 1), (1, 0, 1), (1, 1, 0)],
        'GRBG': [(0, 0, 1), (0, 1, 0), (1, 0, 2), (1, 1, 1)],
        'GBRG': [(0, 0, 1), (0, 1, 2), (1, 0, 0), (1, 1, 1)]
    }
    for row_offset, col_offset, channel_index in mapping[pattern]:
        new_img[row_offset::2, col_offset::2, channel_index] = img[row_offset::2, col_offset::2, channel_index]
    return new_img

def calculate_metrics(raw_img_input: 'numpy.ndarray', new_img_input: 'numpy.ndarray') -> dict:
    """
    Calculates various image quality metrics between a raw and a new image.
    """
    DEFAULT_CROP_SIZE = 5
    CANNY_LOW_THRESHOLD = 100
    CANNY_HIGH_THRESHOLD = 200
    DATA_RANGE = 255

    raw_img = np.copy(raw_img_input)
    new_img = np.copy(new_img_input)
    crop_size = DEFAULT_CROP_SIZE

    if raw_img.shape[:2] != new_img.shape[:2]:
        raise ValueError(f"Input images must have same height/width. Got {raw_img.shape} and {new_img.shape}")
    if raw_img.ndim != 3 or new_img.ndim != 3 or raw_img.shape[2] != 3 or new_img.shape[2] != 3:
        if raw_img.ndim == 2 and new_img.ndim == 2:
            warnings.warn("Grayscale images detected. Adapting for metrics.", UserWarning)
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)
        else:
            warnings.warn(f"Expected RGB images (H, W, 3), got {raw_img.shape} and {new_img.shape}.", UserWarning)

    results = {}
    h, w = raw_img.shape[:2]

    if h <= 2 * crop_size or w <= 2 * crop_size:
        warnings.warn(f"Image too small for cropping. Using full image.", UserWarning)
        crop_raw = raw_img
        crop_new = new_img
    else:
        crop_raw = raw_img[crop_size:-crop_size, crop_size:-crop_size]
        crop_new = new_img[crop_size:-crop_size, crop_size:-crop_size]

    channel_names = ['R', 'G', 'B'] if crop_raw.shape[2] == 3 else ['Gray']
    for c, name in enumerate(channel_names):
        raw_channel = crop_raw[..., c] if crop_raw.ndim == 3 else crop_raw
        new_channel = crop_new[..., c] if crop_new.ndim == 3 else crop_new
        try:
            results[f'PSNR_{name}'] = psnr(raw_channel, new_channel, data_range=DATA_RANGE)
        except Exception as e:
            warnings.warn(f"PSNR_{name} failed: {e}", RuntimeWarning)
            results[f'PSNR_{name}'] = np.nan
        try:
            results[f'SSIM_{name}'] = ssim(raw_channel, new_channel, data_range=DATA_RANGE)
        except Exception as e:
            warnings.warn(f"SSIM_{name} failed: {e}", RuntimeWarning)
            results[f'SSIM_{name}'] = np.nan
        if crop_raw.ndim == 2:
            break

    try:
        results['PSNR_Overall'] = psnr(crop_raw, crop_new, data_range=DATA_RANGE)
    except Exception as e:
        warnings.warn(f"PSNR_Overall failed: {e}", RuntimeWarning)
        results['PSNR_Overall'] = np.nan

    is_multichannel = crop_raw.ndim == 3 and crop_raw.shape[2] > 1
    try:
        if is_multichannel:
            try:
                results['SSIM_Overall'] = ssim(crop_raw, crop_new, data_range=DATA_RANGE, multichannel=True, channel_axis=-1)
            except TypeError:
                warnings.warn("Falling back to older SSIM API.", UserWarning)
                results['SSIM_Overall'] = ssim(crop_raw, crop_new, data_range=DATA_RANGE, multichannel=False, channel_axis=-1)
        else:
            results['SSIM_Overall'] = ssim(crop_raw, crop_new, data_range=DATA_RANGE)
    except Exception as e:
        warnings.warn(f"SSIM_Overall failed: {e}", RuntimeWarning)
        results['SSIM_Overall'] = np.nan

    if raw_img_input.ndim == 3 and raw_img_input.shape[2] == 3:
        try:
            lab_raw = rgb2lab(crop_raw)
            lab_new = rgb2lab(crop_new)
            delta_e_sq = np.sum(np.square(lab_raw - lab_new), axis=2)
            results['Color_MSE_LAB'] = np.mean(delta_e_sq)
        except Exception as e:
            warnings.warn(f"Color_MSE_LAB failed: {e}", RuntimeWarning)
            results['Color_MSE_LAB'] = np.nan
    else:
        results['Color_MSE_LAB'] = np.nan

    try:
        gray_raw = cv2.cvtColor(raw_img_input, cv2.COLOR_RGB2GRAY) if raw_img_input.ndim == 3 else raw_img_input
        gray_new = cv2.cvtColor(new_img_input, cv2.COLOR_RGB2GRAY) if new_img_input.ndim == 3 else new_img_input
        raw_edges = cv2.Canny(gray_raw, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD) > 0
        new_edges = cv2.Canny(gray_new, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD) > 0
        intersection = np.sum(raw_edges & new_edges)
        union = np.sum(raw_edges | new_edges)
        results['Edge_IoU'] = intersection / max(union, 1e-9)
        laplacian_new = cv2.Laplacian(gray_new.astype(np.float64), cv2.CV_64F)
        results['Zipper_StdLap'] = np.std(laplacian_new)
    except Exception as e:
        warnings.warn(f"Edge/Zipper metrics failed: {e}", RuntimeWarning)
        results['Edge_IoU'] = np.nan
        results['Zipper_StdLap'] = np.nan

    return results