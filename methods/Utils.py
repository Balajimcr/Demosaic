import cv2
import numpy as np
import os
import shutil
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.color import rgb2lab
import warnings
import imageio
import numpy as np # Ensure numpy is imported as np
import cv2
import warnings



debug_mode = True
# Bayer pattern slicing dictionary - defined once for all supported patterns
PATTERN_SLICES = {
    'RGGB': {
        'R': (slice(0, None, 2), slice(0, None, 2)),
        'G1': (slice(0, None, 2), slice(1, None, 2)),
        'G2': (slice(1, None, 2), slice(0, None, 2)),
        'B': (slice(1, None, 2), slice(1, None, 2)),
    },
    'BGGR': {
        'B': (slice(0, None, 2), slice(0, None, 2)),
        'G1': (slice(0, None, 2), slice(1, None, 2)),
        'G2': (slice(1, None, 2), slice(0, None, 2)),
        'R': (slice(1, None, 2), slice(1, None, 2)),
    },
    'GRBG': {
        'G1': (slice(0, None, 2), slice(0, None, 2)),
        'R': (slice(0, None, 2), slice(1, None, 2)),
        'B': (slice(1, None, 2), slice(0, None, 2)),
        'G2': (slice(1, None, 2), slice(1, None, 2)),
    },
    'GBRG': {
        'G1': (slice(0, None, 2), slice(0, None, 2)),
        'B': (slice(0, None, 2), slice(1, None, 2)),
        'R': (slice(1, None, 2), slice(0, None, 2)),
        'G2': (slice(1, None, 2), slice(1, None, 2)),
    }
}

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



def calculate_metrics(raw_img_input: np.ndarray, new_img_input: np.ndarray) -> dict:
    """
    Calculates various image quality metrics between a raw (ground truth) and a new (demosaiced) image.

    Args:
        raw_img_input (np.ndarray): The ground truth image (H, W, C). Expected to be RGB.
        new_img_input (np.ndarray): The demosaiced image (H, W, C). Expected to be RGB.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    DEFAULT_CROP_SIZE = 5
    CANNY_LOW_THRESHOLD = 100
    CANNY_HIGH_THRESHOLD = 200
    DATA_RANGE = 255 # Assuming 8-bit images

    # Parameters for CNR calculation
    CNR_VAR_WINDOW_SIZE = 7 # Window size for local variance calculation to find smooth areas
    CNR_VAR_THRESHOLD = 10.0 # Variance threshold to consider a region smooth


    raw_img = np.copy(raw_img_input)
    new_img = np.copy(new_img_input)
    crop_size = DEFAULT_CROP_SIZE

    # --- Input Validation and Adaptation ---
    if raw_img.shape[:2] != new_img.shape[:2]:
        raise ValueError(f"Input images must have same height/width. Got {raw_img.shape[:2]} and {new_img.shape[:2]}")

    # Convert grayscale inputs to RGB if necessary for consistent processing
    if raw_img.ndim == 2:
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
    elif raw_img.ndim == 3 and raw_img.shape[2] == 1:
         raw_img = cv2.cvtColor(raw_img.squeeze(), cv2.COLOR_GRAY2RGB)
    elif raw_img.ndim != 3 or raw_img.shape[2] != 3:
         warnings.warn(f"Unexpected format for raw_img (expected H, W or H, W, 1 or H, W, 3), got {raw_img.shape}. Attempting to process as is if 3 channels.", UserWarning)

    if new_img.ndim == 2:
        new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)
    elif new_img.ndim == 3 and new_img.shape[2] == 1:
         new_img = cv2.cvtColor(new_img.squeeze(), cv2.COLOR_GRAY2RGB)
    elif new_img.ndim != 3 or new_img.shape[2] != 3:
         warnings.warn(f"Unexpected format for new_img (expected H, W or H, W, 1 or H, W, 3), got {new_img.shape}. Attempting to process as is if 3 channels.", UserWarning)

    # Ensure both images are treated as color for subsequent steps if they have 3 channels
    is_color = raw_img.ndim == 3 and raw_img.shape[2] == 3 and new_img.ndim == 3 and new_img.shape[2] == 3


    results = {}
    h, w = raw_img.shape[:2]

    # --- Cropping ---
    if h <= 2 * crop_size or w <= 2 * crop_size:
        warnings.warn(f"Image size ({h}x{w}) too small for {DEFAULT_CROP_SIZE} cropping. Using full image.", UserWarning)
        crop_raw = raw_img
        crop_new = new_img
    else:
        crop_raw = raw_img[crop_size:-crop_size, crop_size:-crop_size]
        crop_new = new_img[crop_size:-crop_size, crop_size:-crop_size]

    # Ensure cropped images are still color if originals were
    is_color_cropped = crop_raw.ndim == 3 and crop_raw.shape[2] == 3 and crop_new.ndim == 3 and crop_new.shape[2] == 3


    # --- PSNR and SSIM per Channel ---
    # Handle potential grayscale after cropping if input wasn't color
    channel_names = ['B', 'G', 'R'] if is_color_cropped else ['Gray'] # OpenCV uses BGR order
    for c, name in enumerate(channel_names):
        # Select channel safely based on dimension
        raw_channel = crop_raw[..., c] if crop_raw.ndim == 3 else crop_raw
        new_channel = crop_new[..., c] if crop_new.ndim == 3 else crop_new

        try:
            # Ensure data types are float for metrics like SSIM if necessary,
            # although data_range helps. psnr/ssim functions usually handle uint8.
            results[f'PSNR_{name}'] = psnr(raw_channel, new_channel, data_range=DATA_RANGE)
        except Exception as e:
            warnings.warn(f"PSNR_{name} failed: {e}", RuntimeWarning)
            results[f'PSNR_{name}'] = np.nan

        try:
            results[f'SSIM_{name}'] = ssim(raw_channel, new_channel, data_range=DATA_RANGE)
        except Exception as e:
            warnings.warn(f"SSIM_{name} failed: {e}", RuntimeWarning)
            results[f'SSIM_{name}'] = np.nan

        if not is_color_cropped: # Only one channel if not color
             break

    # --- Overall PSNR and SSIM ---
    try:
        results['PSNR_Overall'] = psnr(crop_raw, crop_new, data_range=DATA_RANGE)
    except Exception as e:
        warnings.warn(f"PSNR_Overall failed: {e}", RuntimeWarning)
        results['PSNR_Overall'] = np.nan

    try:
        # Use multichannel SSIM if images are color
        if is_color_cropped:
             # Using try-except for older SSIM API compatibility, though channel_axis is standard now
             try:
                 results['SSIM_Overall'] = ssim(crop_raw, crop_new, data_range=DATA_RANGE, channel_axis=-1)
             except TypeError:
                 warnings.warn("Falling back to older SSIM API (multichannel=True).", UserWarning)
                 results['SSIM_Overall'] = ssim(crop_raw, crop_new, data_range=DATA_RANGE, multichannel=True)
        else:
            results['SSIM_Overall'] = ssim(crop_raw, crop_new, data_range=DATA_RANGE)
    except Exception as e:
        warnings.warn(f"SSIM_Overall failed: {e}", RuntimeWarning)
        results['SSIM_Overall'] = np.nan

    # --- Color MSE in LAB ---
    # Only calculate if original inputs were color
    if is_color:
        try:
            # Ensure images are float type for rgb2lab if necessary (skimage expects float in range [0,1] or integer in range [0, DATA_RANGE])
            # Assuming DATA_RANGE is 255 and input is uint8
            lab_raw = rgb2lab(crop_raw.astype(np.float64) / DATA_RANGE if crop_raw.dtype == np.uint8 else crop_raw)
            lab_new = rgb2lab(crop_new.astype(np.float64) / DATA_RANGE if crop_new.dtype == np.uint8 else crop_new)

            delta_e_sq = np.sum(np.square(lab_raw - lab_new), axis=2)
            results['Color_MSE_LAB'] = np.mean(delta_e_sq)
        except Exception as e:
            warnings.warn(f"Color_MSE_LAB failed: {e}", RuntimeWarning)
            results['Color_MSE_LAB'] = np.nan
    else:
        results['Color_MSE_LAB'] = np.nan # Not applicable for grayscale

    # --- CNR (Color Noise Ratio) ---
    # Only calculate if original inputs were color
    if is_color:
        try:
            # Convert cropped GT to grayscale (Y channel)
            # Ensure images are uint8 for cv2.cvtColor if necessary
            crop_raw_gray = cv2.cvtColor(crop_raw.astype(np.uint8), cv2.COLOR_RGB2GRAY) if crop_raw.dtype != np.uint8 else cv2.cvtColor(crop_raw, cv2.COLOR_RGB2GRAY)

            # Calculate local variance map on the GT grayscale image
            # Use cv2.boxFilter for efficiency (mean of squares - square of mean)
            mean_filter = np.ones((CNR_VAR_WINDOW_SIZE, CNR_VAR_WINDOW_SIZE), np.float32) / (CNR_VAR_WINDOW_SIZE * CNR_VAR_WINDOW_SIZE)
            mean_sq = cv2.filter2D(np.square(crop_raw_gray.astype(np.float32)), -1, mean_filter)
            mean = cv2.filter2D(crop_raw_gray.astype(np.float32), -1, mean_filter)
            variance_map = mean_sq - np.square(mean)
            # Clip potential negative values due to float precision
            variance_map = np.maximum(variance_map, 0)

            # Create a mask of smooth regions in the GT
            smooth_mask = variance_map < CNR_VAR_THRESHOLD

            # Check if enough smooth pixels were found
            min_smooth_pixels = CNR_VAR_WINDOW_SIZE * CNR_VAR_WINDOW_SIZE # Require at least one full window of smooth pixels
            if np.sum(smooth_mask) < min_smooth_pixels:
                warnings.warn(f"Not enough smooth regions found for CNR calculation (found {np.sum(smooth_mask)} pixels, need >={min_smooth_pixels}).", RuntimeWarning)
                results['CNR_Cb_Var'] = np.nan
                results['CNR_Cr_Var'] = np.nan
            else:
                # Convert the new (demosaiced) image to YCbCr
                # Ensure images are uint8 for cv2.cvtColor if necessary
                crop_new_ycbcr = cv2.cvtColor(crop_new.astype(np.uint8), cv2.COLOR_RGB2YCrCb) if crop_new.dtype != np.uint8 else cv2.cvtColor(crop_new, cv2.COLOR_RGB2YCrCb)
                # Note: OpenCV uses YCrCb by default, not YCbCr. Channels are Y, Cr, Cb.

                # Extract Cb and Cr channels for pixels within the smooth mask
                # Cb is index 2, Cr is index 1 in OpenCV's YCrCb
                new_cb_smooth = crop_new_ycbcr[:, :, 2][smooth_mask]
                new_cr_smooth = crop_new_ycbcr[:, :, 1][smooth_mask]

                # Calculate variance in the color channels of the demosaiced image over smooth regions
                results['CNR_Cb_Var'] = np.var(new_cb_smooth)
                results['CNR_Cr_Var'] = np.var(new_cr_smooth)

        except Exception as e:
            warnings.warn(f"CNR metrics failed: {e}", RuntimeWarning)
            results['CNR_Cb_Var'] = np.nan
            results['CNR_Cr_Var'] = np.nan
    else:
        results['CNR_Cb_Var'] = np.nan # Not applicable for grayscale
        results['CNR_Cr_Var'] = np.nan # Not applicable for grayscale


    # --- Edge IoU and Zipper StdLap ---
    # Calculate using original image size as edges/zippers can occur anywhere
    # Ensure original images are grayscale for Canny and Laplacian if necessary
    try:
        gray_raw = cv2.cvtColor(raw_img.astype(np.uint8), cv2.COLOR_RGB2GRAY) if is_color else raw_img.astype(np.uint8) # Ensure uint8 for Canny
        gray_new = cv2.cvtColor(new_img.astype(np.uint8), cv2.COLOR_RGB2GRAY) if is_color else new_img.astype(np.uint8)

        # Canny edge detection
        raw_edges = cv2.Canny(gray_raw, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD) > 0
        new_edges = cv2.Canny(gray_new, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD) > 0

        # Edge IoU calculation
        intersection = np.sum(raw_edges & new_edges)
        union = np.sum(raw_edges | new_edges)
        # Avoid division by zero
        results['Edge_IoU'] = intersection / max(union, 1e-9)

        # Zipper StdLap calculation (requires float64 input for Laplacian)
        # Ensure gray_new is float64
        laplacian_new = cv2.Laplacian(gray_new.astype(np.float64), cv2.CV_64F)
        results['Zipper_StdLap'] = np.std(laplacian_new)

    except Exception as e:
        warnings.warn(f"Edge/Zipper metrics failed: {e}", RuntimeWarning)
        results['Edge_IoU'] = np.nan
        results['Zipper_StdLap'] = np.nan


    return results

def update_filename(pattern):
    """Updates the filename prefix for debug image saves."""
    global FileName_DLMMSE
    FileName_DLMMSE = pattern

def save_image(image, stage_name, pattern=None, is_mask=False):
    """Saves an image or mask as a PNG file if debug_mode is enabled."""
    global FileName_DLMMSE
    if not debug_mode:
        return
    foldername = os.path.join("Data/DLMMSE1/")
    os.makedirs(foldername, exist_ok=True)
    
    if is_mask:
        image_to_save = (image * 255).astype(np.uint8)
        filename = f"{FileName_DLMMSE}_{stage_name}"
        if pattern:
            filename = f"{FileName_DLMMSE}_{pattern}_{stage_name}"
        filename += "_mask.png"
    else:
        image_to_save = image.clip(0, 255).astype(np.uint8)
        filename = f"{FileName_DLMMSE}_{stage_name}.png"
    full_path = os.path.join(foldername, filename)
    try:
        if image_to_save.ndim in (2, 3):
            imageio.imwrite(full_path, image_to_save)
        else:
            print(f"Warning: Cannot save image '{filename}' with unexpected shape {image_to_save.shape}")
    except Exception as e:
        print(f"Error saving debug image {full_path}: {e}")

def create_bayer_masks(height, width, pattern):
    """Create binary masks for R, G, and B pixel locations based on Bayer pattern."""
    pattern = pattern.upper()
    if pattern not in PATTERN_SLICES:
        supported = list(PATTERN_SLICES.keys())
        raise ValueError(f"Unsupported Bayer pattern: {pattern}. Supported patterns are: {supported}")
    slices = PATTERN_SLICES[pattern]
    R_mask = np.zeros((height, width), dtype=bool)
    G_mask_01 = np.zeros((height, width), dtype=bool)
    G_mask_10 = np.zeros((height, width), dtype=bool)
    B_mask = np.zeros((height, width), dtype=bool)
    R_mask[slices['R']] = True
    G_mask_01[slices['G1']] = True
    G_mask_10[slices['G2']] = True
    B_mask[slices['B']] = True
    G_mask = G_mask_01 | G_mask_10
    return R_mask, G_mask, G_mask_01, G_mask_10, B_mask