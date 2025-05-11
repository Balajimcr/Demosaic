import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d, uniform_filter
import os
import imageio
import cv2

try:
    from Utils import create_bayer_masks, update_filename, save_image
except ImportError:
    from methods.Utils import create_bayer_masks, update_filename, save_image

# Global configurations
debug_mode = True
FileName_DLMMSE = "DLMMSE_Hybrid"
enable_adaptive_directional_G = True
enable_hybrid_green = True
enable_edge_aware_rb_interpolation = True
enable_adaptive_threshold = True

# Pre-defined kernels (constants)
KERNEL_1D_HV = np.array([-1, 2, 2, 2, -1], dtype=np.float32) / 4.0
GAUSSIAN_FILTER_1D = np.array([4, 9, 15, 23, 26, 23, 15, 9, 4], dtype=np.float32) / 128.0
MEAN_FILTER_1D = np.ones(9, dtype=np.float32) / 9.0
KERNEL_DIAGONAL = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.float32) / 4.0
KERNEL_CROSS = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32) / 4.0

# Pre-computed kernels for common operations
FIXED_KERNEL_G = np.array([0.5, 0, 0.5], dtype=np.float32)
GAUSSIAN_KERNEL = np.array([0.25, 0.5, 0.25], dtype=np.float32)
MEAN_KERNEL = np.array([1/3, 1/3, 1/3], dtype=np.float32)

def compute_gradients(image):
    """Compute gradients efficiently without allocating extra arrays."""
    height, width = image.shape
    grad_x = np.empty_like(image)
    grad_y = np.empty_like(image)
    
    # Compute x gradients (central differences for interior, forward/backward at edges)
    grad_x[:, 1:-1] = (image[:, 2:] - image[:, :-2]) * 0.5
    grad_x[:, 0] = image[:, 1] - image[:, 0]
    grad_x[:, -1] = image[:, -1] - image[:, -2]
    
    # Compute y gradients
    grad_y[1:-1, :] = (image[2:, :] - image[:-2, :]) * 0.5
    grad_y[0, :] = image[1, :] - image[0, :]
    grad_y[-1, :] = image[-1, :] - image[-2, :]
    
    return grad_x, grad_y

def compute_local_gradient_statistics(image, window_size=7):
    """Compute local gradient statistics for adaptive threshold selection."""
    # Use optimized gradient computation
    grad_x, grad_y = compute_gradients(image)
    
    # Compute gradient magnitude in-place
    grad_magnitude = np.sqrt(grad_x*grad_x + grad_y*grad_y)
    
    # Compute local statistics using uniform filter
    local_mean = uniform_filter(grad_magnitude, size=window_size, mode='reflect')
    
    # Compute variance efficiently
    grad_squared = grad_magnitude * grad_magnitude
    local_mean_squared = uniform_filter(grad_squared, size=window_size, mode='reflect')
    local_variance = np.maximum(0, local_mean_squared - local_mean * local_mean)
    local_std = np.sqrt(local_variance)
    
    # Adaptive threshold based on local statistics
    adaptive_threshold = local_mean + 1.5 * local_std
    np.clip(adaptive_threshold, 10.0, 100.0, out=adaptive_threshold)
    
    return adaptive_threshold

def interpolate_green_channel(new_img, R_mask, G_mask_01, G_mask_10, B_mask, debug_info=None):
    """Optimized green channel interpolation."""
    R, G, B = new_img[:, :, 0], new_img[:, :, 1], new_img[:, :, 2]
    S = R + G + B
    
    if debug_info:
        pattern, save = debug_info
        save(S, "1.0_S_sum_of_channels")
    
    # Fixed interpolation
    H_fixed = convolve1d(S, FIXED_KERNEL_G, axis=1, mode='reflect')
    V_fixed = convolve1d(S, FIXED_KERNEL_G, axis=0, mode='reflect')

    if enable_adaptive_directional_G:
        epsilon_g_adapt = 1e-5
        
        # Horizontal adaptive interpolation
        S_pad_h = np.pad(S, ((0, 0), (1, 1)), mode='reflect')
        left = S_pad_h[:, :-2]
        right = S_pad_h[:, 2:]
        
        weight_left = 1.0 / (np.abs(S - left) + epsilon_g_adapt)
        weight_right = 1.0 / (np.abs(S - right) + epsilon_g_adapt)
        sum_weights_h = weight_left + weight_right
        H_adapt = (left * weight_left + right * weight_right) / sum_weights_h
        
        # Vertical adaptive interpolation
        S_pad_v = np.pad(S, ((1, 1), (0, 0)), mode='reflect')
        up = S_pad_v[:-2, :]
        down = S_pad_v[2:, :]
        
        weight_up = 1.0 / (np.abs(S - up) + epsilon_g_adapt)
        weight_down = 1.0 / (np.abs(S - down) + epsilon_g_adapt)
        sum_weights_v = weight_up + weight_down
        V_adapt = (up * weight_up + down * weight_down) / sum_weights_v
        
        if enable_hybrid_green:
            # Adaptive threshold computation
            if enable_adaptive_threshold:
                T = compute_local_gradient_statistics(S)
                if debug_info:
                    save(T, "1.0.5_adaptive_threshold_map")
            else:
                T = 30.0
            
            # Vectorized hybrid blending
            grad_h = np.abs(left - right)
            grad_v = np.abs(up - down)
            w_h = np.clip(grad_h / T, 0, 1)
            w_v = np.clip(grad_v / T, 0, 1)
            
            H = (1 - w_h) * H_fixed + w_h * H_adapt
            V = (1 - w_v) * V_fixed + w_v * V_adapt
            
            if debug_info:
                save(H, "1.1_H_hybrid_interpolation")
                save(V, "1.1_V_hybrid_interpolation")
                save(w_h, "1.1_w_h_blending_weight")
                save(w_v, "1.1_w_v_blending_weight")
        else:
            H = H_adapt
            V = V_adapt
    else:
        H = H_fixed
        V = V_fixed
    
    # Compute deltas
    delta_H = H - S
    delta_V = V - S
    
    if debug_info:
        save(delta_H, "1.1_delta_H_before_sign_flip")
        save(delta_V, "1.1_delta_V_before_sign_flip")
    
    # Sign flip for green locations
    mask_flip = G_mask_01 | G_mask_10
    delta_H[mask_flip] *= -1
    delta_V[mask_flip] *= -1
    
    if debug_info:
        save(delta_H, "1.1_delta_H_after_sign_flip")
        save(delta_V, "1.1_delta_V_after_sign_flip")
    
    # Apply filters
    gaussian_H = convolve1d(delta_H, GAUSSIAN_KERNEL, axis=1, mode='reflect')
    gaussian_V = convolve1d(delta_V, GAUSSIAN_KERNEL, axis=0, mode='reflect')
    
    if debug_info:
        save(gaussian_H, "1.2_gaussian_smoothed_delta_H")
        save(gaussian_V, "1.2_gaussian_smoothed_delta_V")
    
    # Compute mean and variance
    mean_H = convolve1d(gaussian_H, MEAN_KERNEL, axis=1, mode='reflect')
    mean_V = convolve1d(gaussian_V, MEAN_KERNEL, axis=0, mode='reflect')
    
    epsilon_var = 1e-10
    
    # Compute variances efficiently
    diff_H = gaussian_H - mean_H
    diff_V = gaussian_V - mean_V
    var_value_H = convolve1d(diff_H * diff_H, MEAN_KERNEL, axis=1, mode='reflect') + epsilon_var
    var_value_V = convolve1d(diff_V * diff_V, MEAN_KERNEL, axis=0, mode='reflect') + epsilon_var
    
    diff_noise_H = delta_H - gaussian_H
    diff_noise_V = delta_V - gaussian_V
    var_noise_H = convolve1d(diff_noise_H * diff_noise_H, MEAN_KERNEL, axis=1, mode='reflect') + epsilon_var
    var_noise_V = convolve1d(diff_noise_V * diff_noise_V, MEAN_KERNEL, axis=0, mode='reflect') + epsilon_var
    
    if debug_info:
        save(mean_H, "1.3_mean_H")
        save(mean_V, "1.3_mean_V")
        save(var_value_H, "1.3_var_value_H")
        save(var_value_V, "1.3_var_value_V")
        save(var_noise_H, "1.3_var_noise_H")
        save(var_noise_V, "1.3_var_noise_V")
    
    # Signal ratio computation
    signal_ratio_H = var_value_H / (var_noise_H + var_value_H)
    signal_ratio_V = var_value_V / (var_noise_V + var_value_V)
    
    # Refinement
    new_H = mean_H + signal_ratio_H * (delta_H - mean_H)
    new_V = mean_V + signal_ratio_V * (delta_V - mean_V)
    
    if debug_info:
        save(new_H, "1.4_refined_delta_H")
        save(new_V, "1.4_refined_delta_V")
    
    # Compute final weights
    var_x_H = np.abs(var_value_H - var_value_H*var_value_H / (var_value_H + var_noise_H)) + epsilon_var
    var_x_V = np.abs(var_value_V - var_value_V*var_value_V / (var_value_V + var_noise_V)) + epsilon_var
    sum_var_x = var_x_H + var_x_V
    
    w_H = var_x_V / sum_var_x
    w_V = var_x_H / sum_var_x
    final_delta_G = w_H * new_H + w_V * new_V
    
    if debug_info:
        save(var_x_H, "1.5_var_x_H")
        save(var_x_V, "1.5_var_x_V")
        save(w_H, "1.5_weight_H")
        save(w_V, "1.5_weight_V")
        save(final_delta_G, "1.5_final_delta_G")
    
    # Final update
    new_img[R_mask, 1] = new_img[R_mask, 0] + final_delta_G[R_mask]
    new_img[B_mask, 1] = new_img[B_mask, 2] + final_delta_G[B_mask]
    
    if debug_info:
        save(new_img[:, :, 1], "1.6_Final_G_Channel_Interpolated")
    
    return new_img

def interpolate_rb_channels_original(new_img, R_mask, G_mask_combined, B_mask, debug_info=None):
    """Interpolate R and B channels using the original fixed-kernel DLMMSE method."""
    G_interpolated = new_img[:, :, 1].copy()
    
    if debug_info:
        pattern, save = debug_info

    diff_GR = G_interpolated - new_img[:, :, 0]
    diff_GB = G_interpolated - new_img[:, :, 2]
    
    if debug_info:
        save(diff_GR, "2.1_orig_diff_GR_initial_full")
        save(diff_GB, "2.1_orig_diff_GB_initial_full")

    delta_GR_diag = convolve2d(diff_GR, KERNEL_DIAGONAL, mode='same')
    delta_GB_diag = convolve2d(diff_GB, KERNEL_DIAGONAL, mode='same')

    if debug_info:
        save(delta_GR_diag, "2.1_orig_delta_GR_interpolated_diag")
        save(delta_GB_diag, "2.1_orig_delta_GB_interpolated_diag")
    
    new_img[B_mask, 0] = G_interpolated[B_mask] - delta_GR_diag[B_mask]
    new_img[R_mask, 2] = G_interpolated[R_mask] - delta_GB_diag[R_mask]
    
    if debug_info:
        save(new_img[:, :, 0], "2.1_orig_R_after_B_site_interpolation")
        save(new_img[:, :, 2], "2.1_orig_B_after_R_site_interpolation")

    diff_GR_partial = G_interpolated - new_img[:, :, 0]
    diff_GB_partial = G_interpolated - new_img[:, :, 2]

    if debug_info:
        save(diff_GR_partial, "2.2_orig_diff_GR_partial")
        save(diff_GB_partial, "2.2_orig_diff_GB_partial")

    delta_GR_cross = convolve2d(diff_GR_partial, KERNEL_CROSS, mode='same')
    delta_GB_cross = convolve2d(diff_GB_partial, KERNEL_CROSS, mode='same')

    if debug_info:
        save(delta_GR_cross, "2.2_orig_delta_GR_interpolated_cross")
        save(delta_GB_cross, "2.2_orig_delta_GB_interpolated_cross")
        
    new_img[G_mask_combined, 0] = G_interpolated[G_mask_combined] - delta_GR_cross[G_mask_combined]
    new_img[G_mask_combined, 2] = G_interpolated[G_mask_combined] - delta_GB_cross[G_mask_combined]
    
    if debug_info:
        save(new_img[:, :, 0], "2.3_orig_Final_R_Channel")
        save(new_img[:, :, 2], "2.3_orig_Final_B_Channel")
        
    return new_img

def interpolate_rb_channels_enhanced(new_img, R_mask, G_mask_01, G_mask_10, B_mask, debug_info=None):
    """Enhanced edge-aware RB channel interpolation with improved quality."""
    G_interpolated = new_img[:, :, 1].copy()
    epsilon_rb_adapt = 1e-3  # Increased epsilon for stability

    if debug_info:
        pattern, save = debug_info
    
    # Initialize difference arrays - compute differences everywhere
    diff_GR = G_interpolated - new_img[:, :, 0]
    diff_GB = G_interpolated - new_img[:, :, 2]

    if debug_info:
        save(diff_GR, "2.1_enh_diff_GR_initial_full")
        save(diff_GB, "2.1_enh_diff_GB_initial_full")

    # Diagonal interpolation with hybrid approach
    G_pad = np.pad(G_interpolated, ((1, 1), (1, 1)), mode='reflect')
    
    # Compute gradients for adaptive weights
    grad_diag1 = np.abs(G_pad[:-2, :-2] - G_pad[2:, 2:]) + epsilon_rb_adapt
    grad_diag2 = np.abs(G_pad[:-2, 2:] - G_pad[2:, :-2]) + epsilon_rb_adapt
    
    # Use smoother weight function to avoid extreme values
    weight_diag1 = 1.0 / (1.0 + grad_diag1)
    weight_diag2 = 1.0 / (1.0 + grad_diag2)
    
    # Normalize weights
    sum_weights_diag = weight_diag1 + weight_diag2
    weight_diag1 /= sum_weights_diag
    weight_diag2 /= sum_weights_diag

    # Apply convolution-like operation for R channel at B sites
    diff_GR_padded = np.pad(diff_GR, ((1, 1), (1, 1)), mode='reflect')
    
    # Diagonal averaging (similar to KERNEL_DIAGONAL but adaptive)
    interp_dGR_diag1 = (diff_GR_padded[:-2, :-2] + diff_GR_padded[2:, 2:]) / 2.0
    interp_dGR_diag2 = (diff_GR_padded[:-2, 2:] + diff_GR_padded[2:, :-2]) / 2.0
    
    # Weighted combination
    interpolated_dGR = interp_dGR_diag1 * weight_diag1 + interp_dGR_diag2 * weight_diag2
    
    # Fallback to fixed kernel approach in low-gradient areas
    delta_GR_fixed = convolve2d(diff_GR, KERNEL_DIAGONAL, mode='same')
    gradient_strength = (grad_diag1 + grad_diag2) / 2.0
    alpha = np.clip(gradient_strength / 10.0, 0, 1)  # Blend factor
    interpolated_dGR = (1 - alpha) * delta_GR_fixed + alpha * interpolated_dGR
    
    if debug_info: save(interpolated_dGR, "2.2_enh_interpolated_dGR_for_B_sites")
    new_img[B_mask, 0] = G_interpolated[B_mask] - interpolated_dGR[B_mask]
    if debug_info: save(new_img[:, :, 0], "2.3_enh_R_channel_after_B_site_interpolation")

    # Similar approach for B channel at R sites
    diff_GB_padded = np.pad(diff_GB, ((1, 1), (1, 1)), mode='reflect')
    
    interp_dGB_diag1 = (diff_GB_padded[:-2, :-2] + diff_GB_padded[2:, 2:]) / 2.0
    interp_dGB_diag2 = (diff_GB_padded[:-2, 2:] + diff_GB_padded[2:, :-2]) / 2.0
    
    interpolated_dGB = interp_dGB_diag1 * weight_diag1 + interp_dGB_diag2 * weight_diag2
    
    # Fallback blend
    delta_GB_fixed = convolve2d(diff_GB, KERNEL_DIAGONAL, mode='same')
    interpolated_dGB = (1 - alpha) * delta_GB_fixed + alpha * interpolated_dGB

    if debug_info: save(interpolated_dGB, "2.4_enh_interpolated_dGB_for_R_sites")
    new_img[R_mask, 2] = G_interpolated[R_mask] - interpolated_dGB[R_mask]
    if debug_info: save(new_img[:, :, 2], "2.5_enh_B_channel_after_R_site_interpolation")

    # Cross interpolation at G sites
    diff_GR_partial = G_interpolated - new_img[:, :, 0]
    diff_GB_partial = G_interpolated - new_img[:, :, 2]
    
    if debug_info:
        save(diff_GR_partial, "2.6_enh_diff_GR_partial_after_diag_interp")
        save(diff_GB_partial, "2.6_enh_diff_GB_partial_after_diag_interp")

    # Use fixed kernel as base
    delta_GR_cross_fixed = convolve2d(diff_GR_partial, KERNEL_CROSS, mode='same')
    delta_GB_cross_fixed = convolve2d(diff_GB_partial, KERNEL_CROSS, mode='same')
    
    # Adaptive refinement only in high-contrast areas
    G_grad_h = np.abs(np.gradient(G_interpolated, axis=1))
    G_grad_v = np.abs(np.gradient(G_interpolated, axis=0))
    edge_strength = np.sqrt(G_grad_h**2 + G_grad_v**2)
    
    # Blend adaptive and fixed based on edge strength
    beta = np.clip(edge_strength / 20.0, 0, 0.5)  # Limited adaptive contribution
    
    # Simplified adaptive cross interpolation
    G_mask_combined = G_mask_01 | G_mask_10
    
    new_img[G_mask_combined, 0] = G_interpolated[G_mask_combined] - delta_GR_cross_fixed[G_mask_combined]
    new_img[G_mask_combined, 2] = G_interpolated[G_mask_combined] - delta_GB_cross_fixed[G_mask_combined]
    
    if debug_info:
        save(new_img[:, :, 0], "2.8_enh_Final_R_Channel")
        save(new_img[:, :, 2], "2.8_enh_Final_B_Channel")
        
    return new_img

def run(img, pattern='RGGB'):
    """DLMMSE demosaicing function with optimized operations."""
    height, width, _ = img.shape
    
    def save_debug_image_local(image_data, stage_name_local, is_mask_local=False):
        update_filename(pattern)
        save_image(image_data, stage_name_local, pattern=pattern.upper(), is_mask=is_mask_local)
    
    debug_info_tuple = (pattern, save_debug_image_local) if debug_mode else None
    
    # Use float32 for better memory efficiency
    new_img_float = img.astype(np.float32)
    
    if debug_mode and debug_info_tuple:
        _, save_dbg_func = debug_info_tuple
        save_dbg_func(new_img_float, "0.0_input_bayer_img")
    
    R_mask, G_mask_combined, G_mask_01, G_mask_10, B_mask = create_bayer_masks(height, width, pattern)
    
    if debug_mode and debug_info_tuple:
        _, save_dbg_func = debug_info_tuple
        save_dbg_func(R_mask, "0.1_R_mask", is_mask_local=True)
        save_dbg_func(G_mask_01, "0.1_G_mask_01_type1", is_mask_local=True)
        save_dbg_func(G_mask_10, "0.1_G_mask_10_type2", is_mask_local=True)
        save_dbg_func(G_mask_combined, "0.1_G_mask_combined", is_mask_local=True)
        save_dbg_func(new_img_float[:, :, 0] * R_mask, "0.2_initial_R_channel_masked")
        save_dbg_func(new_img_float[:, :, 1] * G_mask_combined, "0.2_initial_G_channel_masked")
        save_dbg_func(new_img_float[:, :, 2] * B_mask, "0.2_initial_B_channel_masked")
    
    new_img_float = interpolate_green_channel(new_img_float, R_mask, G_mask_01, G_mask_10, B_mask, debug_info_tuple)
    
    if enable_edge_aware_rb_interpolation:
        if debug_mode: print("Using Enhanced Edge-Aware RB Interpolation (Optimized).")
        new_img_float = interpolate_rb_channels_enhanced(new_img_float, R_mask, G_mask_01, G_mask_10, B_mask, debug_info_tuple)
    else:
        if debug_mode: print("Using Original Fixed-Kernel RB Interpolation.")
        new_img_float = interpolate_rb_channels_original(new_img_float, R_mask, G_mask_combined, B_mask, debug_info_tuple)
    
    # Efficient final conversion
    final_demosaiced_img = np.clip(new_img_float + 0.5, 0, 255).astype(np.uint8)
    
    if debug_mode and debug_info_tuple:
        _, save_dbg_func = debug_info_tuple
        rb_mode_suffix = "_RBenhanced" if enable_edge_aware_rb_interpolation else "_RBoriginal"
        
        features_str = ""
        if enable_adaptive_threshold:
            features_str += "_AdaptThresh"
            
        save_dbg_func(final_demosaiced_img, f"3.0_Final_Demosaiced_Image{rb_mode_suffix}{features_str}")
    
    return final_demosaiced_img

def test():
    """Runs a self-contained test of the DLMMSE demosaicing algorithm."""
    print(f"Starting self-test for DLMMSE (Optimized Version)...")
    print(f"Edge-Aware RB Interpolation ENABLED: {enable_edge_aware_rb_interpolation}")
    
    print(f"Novel features enabled:")
    print(f"   - Adaptive Threshold: {enable_adaptive_threshold}")
    
    global FileName_DLMMSE
    original_global_filename_prefix = FileName_DLMMSE
    
    features_str = ""
    if enable_adaptive_threshold:
        features_str += "AdaptThresh_"
    
    rb_mode_str = "EdgeAwareRB" if enable_edge_aware_rb_interpolation else "OriginalRB"
    FileName_DLMMSE = f"UnitTest_DLMMSE_{features_str}{rb_mode_str}_Optimized"

    input_image_filename = "RGGB_0.0_input_bayer_img.png"
    base_data_dir = "Data"
    input_subdir = "DLMMSE1"
    input_image_path = os.path.join(base_data_dir, input_subdir, input_image_filename)
    test_pattern = 'RGGB'
    
    print(f"Attempting to load Bayer image from: {input_image_path}")
    
    try:
        img_bgr_uint8 = cv2.imread(input_image_path)
        if img_bgr_uint8 is None:
            alt_input_path = os.path.join("..", base_data_dir, input_subdir, input_image_filename)
            img_bgr_uint8 = cv2.imread(alt_input_path)
            if img_bgr_uint8 is None:
                raise FileNotFoundError(f"Image file not found at {input_image_path} or {alt_input_path}")
            input_image_path = alt_input_path
        img_rgb_uint8 = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        FileName_DLMMSE = original_global_filename_prefix
        return
    except Exception as e:
        print(f"An critical error occurred during image loading: {e}")
        import traceback
        traceback.print_exc()
        FileName_DLMMSE = original_global_filename_prefix
        return
    
    test_output_main_dir = os.path.join(base_data_dir, f"UnitTest_DLMMSE_{features_str}{rb_mode_str}_Optimized_Output")
    os.makedirs(test_output_main_dir, exist_ok=True)
    print(f"Test output (final image) will be saved in: {test_output_main_dir}")
    if debug_mode:
        print(f"Debug images (if enabled) will be in a subfolder like: Data/{FileName_DLMMSE}_{test_pattern}/")

    print(f"Running OPTIMIZED DLMMSE algorithm (RB mode: {'Enhanced' if enable_edge_aware_rb_interpolation else 'Original'}) on loaded image (pattern: {test_pattern})...")
    
    try:
        demosaiced_img = run(img_rgb_uint8, pattern=test_pattern)
        base_input_fn = os.path.basename(input_image_path)
        demosaiced_output_filename = f"demosaiced_{features_str}{rb_mode_str}_output_of_{base_input_fn}"
        demosaiced_output_path = os.path.join(test_output_main_dir, demosaiced_output_filename)
        imageio.imwrite(demosaiced_output_path, demosaiced_img)
        print(f"Successfully saved final demosaiced image to {demosaiced_output_path}")

    except Exception as e:
        print(f"An error occurred during the DLMMSE test run: {e}")
        import traceback
        traceback.print_exc()
    finally:
        FileName_DLMMSE = original_global_filename_prefix
    
    print(f"Self-test for DLMMSE (RB mode: {'Enhanced' if enable_edge_aware_rb_interpolation else 'Original'}) finished.")

def test_with_timing():
    """Test with performance timing comparison."""
    import time
    
    # Load test image
    input_image_filename = "RGGB_0.0_input_bayer_img.png"
    base_data_dir = "Data"
    input_subdir = "DLMMSE1"
    input_image_path = os.path.join(base_data_dir, input_subdir, input_image_filename)
    
    try:
        img_bgr_uint8 = cv2.imread(input_image_path)
        if img_bgr_uint8 is None:
            alt_input_path = os.path.join("..", base_data_dir, input_subdir, input_image_filename)
            img_bgr_uint8 = cv2.imread(alt_input_path)
            if img_bgr_uint8 is None:
                raise FileNotFoundError(f"Image file not found")
        img_rgb_uint8 = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB)
    except:
        print("Could not load test image for timing")
        return
    
    # Time the optimized version
    n_runs = 5
    times = []
    
    print(f"Running {n_runs} iterations for timing...")
    for i in range(n_runs):
        start_time = time.time()
        demosaiced_img = run(img_rgb_uint8, pattern='RGGB')
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Run {i+1}: {times[-1]:.3f} seconds")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"\nAverage time: {avg_time:.3f} ± {std_time:.3f} seconds")

if __name__ == "__main__":
    print("--- Testing Optimized DLMMSE ---")
    print("1. Testing with Original DLMMSE (no novel features)")
    enable_adaptive_threshold = False
    enable_edge_aware_rb_interpolation = False
    test()
    
    print("\n2. Testing with Only Adaptive Threshold")
    enable_adaptive_threshold = True
    enable_edge_aware_rb_interpolation = True
    test()
    
    print("\n3. Performance Timing Test")
    test_with_timing()