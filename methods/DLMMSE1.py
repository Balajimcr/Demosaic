import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d
import os
import imageio
import cv2

# Global configurations
debug_mode = True
FileName_DLMMSE = "DLMMSE"  # Default filename prefix

# New feature flags:
#   enable_adaptive_directional enables adaptive directional estimation for interpolation.
enable_adaptive_directional = True

# Pre-define commonly used kernels
KERNEL_1D_HV = np.array([-1, 2, 2, 2, -1]) / 4.0
GAUSSIAN_FILTER_1D = np.array([4, 9, 15, 23, 26, 23, 15, 9, 4]) / 128.0  # Pre-normalized
MEAN_FILTER_1D = np.ones(9) / 9.0
KERNEL_DIAGONAL = np.array([[1, 0, 1],
                            [0, 0, 0],
                            [1, 0, 1]]) / 4.0
KERNEL_CROSS = np.array([[0, 1, 0],
                         [1, 0, 1],
                         [0, 1, 0]]) / 4.0

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

def update_filename(pattern):
    """Updates the filename prefix for debug image saves."""
    global FileName_DLMMSE
    FileName_DLMMSE = pattern

def save_image(image, stage_name, pattern=None, is_mask=False):
    """Saves an image or mask as a PNG file if debug_mode is enabled."""
    if not debug_mode:
        return

    # Create debug directory if it doesn't exist
    foldername = os.path.join("Data/DLMMSE1/")
    os.makedirs(foldername, exist_ok=True)

    # Prepare the image for saving
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
        # Handle different image dimensions
        if image_to_save.ndim in (2, 3):
            imageio.imwrite(full_path, image_to_save)
        else:
            print(f"Warning: Cannot save image '{filename}' with unexpected shape {image_to_save.shape}")
    except Exception as e:
        print(f"Error saving debug image {full_path}: {e}")

def create_bayer_masks(height, width, pattern):
    """Create binary masks for R, G, and B pixel locations based on Bayer pattern."""
    # Validate pattern
    pattern = pattern.upper()
    if pattern not in PATTERN_SLICES:
        supported = list(PATTERN_SLICES.keys())
        raise ValueError(f"Unsupported Bayer pattern: {pattern}. Supported patterns are: {supported}")
    
    # Get slices for this pattern
    slices = PATTERN_SLICES[pattern]
    
    # Create binary masks
    R_mask = np.zeros((height, width), dtype=bool)
    G_mask_01 = np.zeros((height, width), dtype=bool)
    G_mask_10 = np.zeros((height, width), dtype=bool)
    B_mask = np.zeros((height, width), dtype=bool)
    
    # Assign masks using the slicing dictionary
    R_mask[slices['R']] = True
    G_mask_01[slices['G1']] = True
    G_mask_10[slices['G2']] = True
    B_mask[slices['B']] = True
    
    # Combined G mask
    G_mask = G_mask_01 | G_mask_10
    
    return R_mask, G_mask, G_mask_01, G_mask_10, B_mask

def interpolate_green_channel(new_img, R_mask, G_mask_01, G_mask_10, B_mask, debug_info=None):
    """Interpolate the G channel at R and B positions."""
    # Use fixed kernels for Gaussian and mean filtering within this function
    from scipy.ndimage import convolve1d
    
    R, G, B = new_img[:, :, 0], new_img[:, :, 1], new_img[:, :, 2]
    S = R + G + B  # Sum of channels (sparse at initial stage)
    
    if debug_info:
        pattern, save = debug_info
        save(S, "1.0_S_sum_of_channels")
    
    # 1.1: Directional interpolation: either adaptive or fixed
    if enable_adaptive_directional:
        epsilon = 1e-5
        # --- Adaptive Horizontal Interpolation ---
        # Pad the image horizontally (with reflect mode) so that left/right neighbors are available.
        padded_S_horiz = np.pad(S, ((0, 0), (1, 1)), mode='reflect')
        left = padded_S_horiz[:, :-2]
        right = padded_S_horiz[:, 2:]
        # Calculate weights based on the absolute difference with the central pixel
        weight_left = 1.0 / (np.abs(S - left) + epsilon)
        weight_right = 1.0 / (np.abs(S - right) + epsilon)
        sum_weights = weight_left + weight_right
        H = (left * weight_left + right * weight_right) / sum_weights
        
        # --- Adaptive Vertical Interpolation ---
        padded_S_vert = np.pad(S, ((1, 1), (0, 0)), mode='reflect')
        up = padded_S_vert[:-2, :]
        down = padded_S_vert[2:, :]
        weight_up = 1.0 / (np.abs(S - up) + epsilon)
        weight_down = 1.0 / (np.abs(S - down) + epsilon)
        sum_weights_v = weight_up + weight_down
        V = (up * weight_up + down * weight_down) / sum_weights_v
        
        if debug_info:
            pattern, save = debug_info
            save(H, "1.1_H_adaptive_interpolation")
            save(V, "1.1_V_adaptive_interpolation")
    else:
        # Fixed directional interpolation using predefined kernel [1/2, 0, 1/2]
        fixed_kernel = np.array([1/2, 0, 1/2])
        H = convolve1d(S, fixed_kernel, axis=1, mode='reflect')
        V = convolve1d(S, fixed_kernel, axis=0, mode='reflect')
        
        if debug_info:
            pattern, save = debug_info
            save(H, "1.1_H_fixed_interpolation")
            save(V, "1.1_V_fixed_interpolation")
    
    # Compute delta from interpolation
    delta_H = H - S
    delta_V = V - S
    
    if debug_info:
        pattern, save = debug_info
        save(delta_H, "1.1_delta_H_before_sign_flip")
        save(delta_V, "1.1_delta_V_before_sign_flip")
    
    # Flip sign at G pixel locations
    delta_H[G_mask_01] = -delta_H[G_mask_01]
    delta_H[G_mask_10] = -delta_H[G_mask_10]
    delta_V[G_mask_01] = -delta_V[G_mask_01]
    delta_V[G_mask_10] = -delta_V[G_mask_10]
    
    if debug_info:
        pattern, save = debug_info
        save(delta_H, "1.1_delta_H_after_sign_flip")
        save(delta_V, "1.1_delta_V_after_sign_flip")
    
    # 1.2: Apply Gaussian filtering to deltas
    gaussian_filter_kernel = np.array([1/4, 1/2, 1/4])
    gaussian_H = convolve1d(delta_H, gaussian_filter_kernel, axis=1, mode='reflect')
    gaussian_V = convolve1d(delta_V, gaussian_filter_kernel, axis=0, mode='reflect')
    
    if debug_info:
        pattern, save = debug_info
        save(gaussian_H, "1.2_gaussian_smoothed_delta_H")
        save(gaussian_V, "1.2_gaussian_smoothed_delta_V")
    
    # 1.3: Calculate statistics (mean and variance)
    mean_filter_kernel = np.array([1/3, 1/3, 1/3])
    mean_H = convolve1d(gaussian_H, mean_filter_kernel, axis=1, mode='reflect')
    mean_V = convolve1d(gaussian_V, mean_filter_kernel, axis=0, mode='reflect')
    
    epsilon = 1e-10  # Avoid division by zero
    var_value_H = convolve1d(np.square(gaussian_H - mean_H), mean_filter_kernel, axis=1, mode='reflect') + epsilon
    var_value_V = convolve1d(np.square(gaussian_V - mean_V), mean_filter_kernel, axis=0, mode='reflect') + epsilon
    
    var_noise_H = convolve1d(np.square(delta_H - gaussian_H), mean_filter_kernel, axis=1, mode='reflect') + epsilon
    var_noise_V = convolve1d(np.square(delta_V - gaussian_V), mean_filter_kernel, axis=0, mode='reflect') + epsilon
    
    if debug_info:
        pattern, save = debug_info
        save(mean_H, "1.3_mean_H")
        save(mean_V, "1.3_mean_V")
        save(var_value_H, "1.3_var_value_H")
        save(var_value_V, "1.3_var_value_V")
        save(var_noise_H, "1.3_var_noise_H")
        save(var_noise_V, "1.3_var_noise_V")
    
    # 1.4: Refine delta maps (Wiener-like filtering)
    signal_ratio_H = var_value_H / (var_noise_H + var_value_H)
    signal_ratio_V = var_value_V / (var_noise_V + var_value_V)
    
    new_H = mean_H + signal_ratio_H * (delta_H - mean_H)
    new_V = mean_V + signal_ratio_V * (delta_V - mean_V)
    
    if debug_info:
        pattern, save = debug_info
        save(new_H, "1.4_refined_delta_H")
        save(new_V, "1.4_refined_delta_V")
    
    # 1.5: Combine horizontal and vertical refined delta estimates
    var_x_H = np.abs(var_value_H - var_value_H**2 / (var_value_H + var_noise_H)) + epsilon
    var_x_V = np.abs(var_value_V - var_value_V**2 / (var_value_V + var_noise_V)) + epsilon
    sum_var_x = var_x_H + var_x_V
    w_H = var_x_V / sum_var_x
    w_V = var_x_H / sum_var_x
    final_delta_G = w_H * new_H + w_V * new_V
    
    if debug_info:
        pattern, save = debug_info
        save(var_x_H, "1.5_var_x_H")
        save(var_x_V, "1.5_var_x_V")
        save(w_H, "1.5_weight_H")
        save(w_V, "1.5_weight_V")
        save(final_delta_G, "1.5_final_delta_G")
    
    # 1.6: Update G values at R and B locations
    new_img[R_mask, 1] = new_img[R_mask, 0] + final_delta_G[R_mask]
    new_img[B_mask, 1] = new_img[B_mask, 2] + final_delta_G[B_mask]
    
    if debug_info:
        pattern, save = debug_info
        save(new_img[:, :, 1], "1.6_G_interpolated_at_RB")
        save(new_img[:, :, 1], "1.6_Final_G_Channel")
    
    return new_img

def interpolate_rb_channels(new_img, R_mask, G_mask, B_mask, debug_info=None):
    """Interpolate the R and B channels at missing locations."""
    G_interpolated = new_img[:, :, 1]
    R_original = new_img[:, :, 0]
    B_original = new_img[:, :, 2]
    
    # 2.1: Calculate initial color differences
    diff_GR = G_interpolated - R_original
    diff_GB = G_interpolated - B_original
    
    if debug_info:
        pattern, save = debug_info
        save(diff_GR, "2.1_diff_GR_initial")
        save(diff_GB, "2.1_diff_GB_initial")
    
    # Interpolate the differences using the diagonal kernel for RB at complementary locations
    delta_GR_diag = convolve2d(diff_GR, KERNEL_DIAGONAL, mode='same')
    delta_GB_diag = convolve2d(diff_GB, KERNEL_DIAGONAL, mode='same')
    
    if debug_info:
        pattern, save = debug_info
        save(delta_GR_diag, "2.1_delta_GR_interpolated_diag")
        save(delta_GB_diag, "2.1_delta_GB_interpolated_diag")
    
    # Update R and B values at complementary locations
    new_img[B_mask, 0] = G_interpolated[B_mask] - delta_GR_diag[B_mask]
    new_img[R_mask, 2] = G_interpolated[R_mask] - delta_GB_diag[R_mask]
    
    if debug_info:
        pattern, save = debug_info
        save(new_img[:, :, 0], "2.1_R_after_interpolation_at_B")
        save(new_img[:, :, 2], "2.1_B_after_interpolation_at_R")
    
    # 2.2: Recalculate differences for G locations
    diff_GR_partial = G_interpolated - new_img[:, :, 0]
    diff_GB_partial = G_interpolated - new_img[:, :, 2]
    
    if debug_info:
        pattern, save = debug_info
        save(diff_GR_partial, "2.2_diff_GR_partial")
        save(diff_GB_partial, "2.2_diff_GB_partial")
    
    # Interpolate the differences using the cross kernel for G locations
    delta_GR_cross = convolve2d(diff_GR_partial, KERNEL_CROSS, mode='same')
    delta_GB_cross = convolve2d(diff_GB_partial, KERNEL_CROSS, mode='same')
    
    if debug_info:
        pattern, save = debug_info
        save(delta_GR_cross, "2.2_delta_GR_interpolated_cross")
        save(delta_GB_cross, "2.2_delta_GB_interpolated_cross")
    
    # Fill in R and B at G locations
    new_img[G_mask, 0] = G_interpolated[G_mask] - delta_GR_cross[G_mask]
    new_img[G_mask, 2] = G_interpolated[G_mask] - delta_GB_cross[G_mask]
    
    if debug_info:
        pattern, save = debug_info
        save(new_img[:, :, 0], "2.3_Final_R_Channel")
        save(new_img[:, :, 2], "2.3_Final_B_Channel")
    
    return new_img

def run(img, pattern='RGGB'):
    """
    DLMMSE demosaicing function for different Bayer patterns.
    
    Args:
        img (numpy.ndarray): Input 3-channel image with a Bayer mosaic pattern.
        pattern (str): The Bayer pattern ('RGGB', 'BGGR', 'GRBG', 'GBRG').
        
    Returns:
        numpy.ndarray: Demosaiced 3-channel (RGB) image.
    """
    height, width, _ = img.shape
    
    # Update filename for debug output
    update_filename(pattern)
    
    # Create a save function closure for debug images
    def save_debug_image(image, stage_name, is_mask=False):
        save_image(image, stage_name, pattern=pattern.upper(), is_mask=is_mask)
    
    debug_info = (pattern, save_debug_image) if debug_mode else None
    
    # Convert to float for processing
    new_img = np.copy(img).astype(float)
    
    if debug_mode:
        save_debug_image(new_img, "0.0_input_bayer_img")
    
    # Create masks for each color based on the pattern
    R_mask, G_mask, G_mask_01, G_mask_10, B_mask = create_bayer_masks(height, width, pattern)
    
    if debug_mode:
        save_debug_image(R_mask, "R_mask", is_mask=True)
        save_debug_image(G_mask_01, "G_mask_type1", is_mask=True)
        save_debug_image(G_mask_10, "G_mask_type2", is_mask=True)
        save_debug_image(G_mask, "G_mask_combined", is_mask=True)
        save_debug_image(B_mask, "B_mask", is_mask=True)
        save_debug_image(new_img[:, :, 0], "0.2_initial_R")
        save_debug_image(new_img[:, :, 1], "0.2_initial_G")
        save_debug_image(new_img[:, :, 2], "0.2_initial_B")
    
    # Step 1: Interpolate the G channel
    new_img = interpolate_green_channel(new_img, R_mask, G_mask_01, G_mask_10, B_mask, debug_info)
    
    # Step 2: Interpolate the R and B channels
    new_img = interpolate_rb_channels(new_img, R_mask, G_mask, B_mask, debug_info)
    
    # Final image conversion
    final_demosaiced_img = (new_img + 0.5).clip(0, 255.5).astype(np.uint8)
    
    if debug_mode:
        save_debug_image(final_demosaiced_img, "3.0_Final_Demosaiced_Image")
    
    return final_demosaiced_img

def test():
    """
    Runs a self-contained test of the DLMMSE demosaicing algorithm
    using a specific pre-saved Bayer image file.
    Saves the demosaiced output to a dedicated test directory.
    """
    print("Starting self-test for DLMMSE with pre-saved image...")
    
    # Define the path to the pre-saved Bayer image
    input_image_path = os.path.join("Data", "DLMMSE1", "RGGB_0.0_input_bayer_img.png")
    test_pattern = 'RGGB'
    
    print(f"Attempting to load Bayer image from: {input_image_path}")
    
    try:
        # Load the image using cv2 (loads as BGR uint8 by default)
        img_bgr_uint8 = cv2.imread(input_image_path)
        if img_bgr_uint8 is None:
            raise FileNotFoundError(f"Image file not found or failed to load: {input_image_path}")
        
        # Convert BGR to RGB. The run function and masks assume RGB channel order (R=0, G=1, B=2)
        img_rgb_uint8 = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure the required input Bayer image exists at the correct path.")
        return  # Exit if file loading fails
    except Exception as e:
        print(f"An error occurred during image loading: {e}")
        import traceback
        traceback.print_exc()
        return  # Exit on other loading errors
    
    # Set a specific filename prefix for debug output during this test run
    global FileName_DLMMSE
    original_filename_prefix = FileName_DLMMSE  # Store original
    
    # Use a prefix based on the loaded input filename for clarity
    FileName_DLMMSE = f"test_loaded_{os.path.splitext(os.path.basename(input_image_path))[0]}"
    
    # Define test output directory, separate from general debug steps
    test_output_dir = os.path.join("Data", "DLMMSE1", "test_output_loaded")
    os.makedirs(test_output_dir, exist_ok=True)
    
    print(f"Saving test results to: {test_output_dir}")
    
    # Run the demosaicing algorithm
    print(f"Running DLMMSE algorithm on loaded image (pattern: {test_pattern})...")
    
    try:
        # Pass the loaded uint8 RGB image to the run function
        # The run function handles the conversion to float internally
        demosaiced_img = run(img_rgb_uint8, pattern=test_pattern)
        
        # Save the demosaiced output image
        base_input_filename = os.path.basename(input_image_path)
        demosaiced_output_filename = f"demosaiced_output_of_{base_input_filename}"
        demosaiced_output_path = os.path.join(test_output_dir, demosaiced_output_filename)
        
        print(f"Saving demosaiced output to {demosaiced_output_path}")
        
        # The run function returns a uint8 image, ready to save
        imageio.imwrite(demosaiced_output_path, demosaiced_img)
    except Exception as e:
        print(f"An error occurred during the DLMMSE test run: {e}")
        import traceback
        traceback.print_exc()  # Print traceback for debugging
    finally:
        # Always restore the original filename prefix
        FileName_DLMMSE = original_filename_prefix
    
    print("Self-test for DLMMSE with pre-saved image finished.")
    print(f"Check '{test_output_dir}' for the demosaiced output image.")
    
    if debug_mode:
        print(f"Check 'Data/DLMMSE1' for intermediate debug images (if enabled).")

if __name__ == "__main__":
    test()