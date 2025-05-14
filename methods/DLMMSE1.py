"""DLMMSE demosaicing algorithm implementation with optimized operations."""

import os
import numpy as np
import cv2
import imageio
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d, uniform_filter

try:
    from Utils import create_bayer_masks, update_filename, save_image
except ImportError:
    from methods.Utils import create_bayer_masks, update_filename, save_image

# Global configurations
debug_mode = False
FileName_DLMMSE = "DLMMSE_Hybrid"
enable_adaptive_directional_G = True
enable_hybrid_green = True
enable_edge_aware_rb_interpolation = True
enable_adaptive_threshold = True
enable_green_channel_GBTF_interpolation = True

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

# Constants
WEIGHT_EPSILON = 1e-10
EPSILON_RB_ADAPT = 1e-3


def debug_save(debug_info, data, name):
    """Save debug image if debug_info is provided."""
    if debug_info:
        _, save = debug_info
        save(data, name)

import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d

# Pre-defined kernels (constants)
KERNEL_1D_HV = np.array([-1, 2, 2, 2, -1], dtype=np.float32) / 4.0
WEIGHT_EPSILON = 1e-10

def debug_save(debug_info, data, name):
    """Save debug image if debug_info is provided."""
    if debug_info:
        _, save = debug_info
        save(data, name)

def create_gbtf_weight_kernels():
    """Create GBTF weight kernels."""
    kernels = {}
    
    # West kernel
    kernels['W_W'] = np.zeros((9, 9), dtype=np.float32)
    kernels['W_W'][2:7, 0:5] = 1.0
    
    # East kernel
    kernels['W_E'] = np.zeros((9, 9), dtype=np.float32)
    kernels['W_E'][2:7, 4:9] = 1.0
    
    # North kernel
    kernels['W_N'] = np.zeros((9, 9), dtype=np.float32)
    kernels['W_N'][0:5, 2:7] = 1.0
    
    # South kernel
    kernels['W_S'] = np.zeros((9, 9), dtype=np.float32)
    kernels['W_S'][4:9, 2:7] = 1.0
    
    return kernels

def process_gbtf_weights(D_H, D_V, weight_kernels):
    """Process GBTF directional weights with minimal modifications."""
    # Compute weights (original GBTF approach)
    W_W = convolve2d(D_H, weight_kernels['W_W'], mode='same', boundary='fill', fillvalue=0.0)
    W_E = convolve2d(D_H, weight_kernels['W_E'], mode='same', boundary='fill', fillvalue=0.0)
    W_N = convolve2d(D_V, weight_kernels['W_N'], mode='same', boundary='fill', fillvalue=0.0)
    W_S = convolve2d(D_V, weight_kernels['W_S'], mode='same', boundary='fill', fillvalue=0.0)
    
    # Process weights - slight modification for robustness in smooth regions
    for W in [W_W, W_E, W_N, W_S]:
        # Add small adaptive epsilon based on local gradient magnitude
        local_eps = WEIGHT_EPSILON * (1 + 0.1 * np.mean(D_H + D_V))
        W[W == 0] = local_eps
        W[:] = 1.0 / np.square(W + local_eps)
    
    W_T = W_W + W_E + W_N + W_S
    
    return W_W, W_E, W_N, W_S, W_T

def compute_gbtf_delta(delta_pattern, pattern_indices, weight_data):
    """Compute GBTF delta for specific pattern."""
    W_N, W_S, W_E, W_W, W_T = weight_data
    
    # Prepare delta arrays
    delta_H_pattern = np.zeros_like(delta_pattern[0])
    delta_V_pattern = np.zeros_like(delta_pattern[1])
    
    if pattern_indices[0] == 0:  # c=0 pattern
        delta_H_pattern[0::2, :] = delta_pattern[0][0::2, :]
        delta_V_pattern[:, 0::2] = delta_pattern[1][:, 0::2]
    else:  # c=1 pattern
        delta_H_pattern[1::2, :] = delta_pattern[0][1::2, :]
        delta_V_pattern[:, 1::2] = delta_pattern[1][:, 1::2]
    
    # Define F coefficients kernels
    F_COEFFS_1D = np.full(5, 0.2, dtype=np.float32)
    KERNEL_F_FORWARD_1D = np.zeros(9, dtype=np.float32)
    KERNEL_F_FORWARD_1D[0:5] = F_COEFFS_1D
    KERNEL_F_BACKWARD_1D = np.zeros(9, dtype=np.float32)
    KERNEL_F_BACKWARD_1D[4:9] = F_COEFFS_1D[::-1]
    
    # Convolve with F kernels
    V1_N = convolve1d(delta_V_pattern, KERNEL_F_FORWARD_1D, axis=0, mode='reflect')
    V2_S = convolve1d(delta_V_pattern, KERNEL_F_BACKWARD_1D, axis=0, mode='reflect')
    V3_E = convolve1d(delta_H_pattern, KERNEL_F_FORWARD_1D, axis=1, mode='reflect')
    V4_W = convolve1d(delta_H_pattern, KERNEL_F_BACKWARD_1D, axis=1, mode='reflect')
    
    return (V1_N * W_N + V2_S * W_S + V3_E * W_E + V4_W * W_W) / W_T

def interpolate_green_channel(new_img, R_mask, G_mask_01, G_mask_10, B_mask, debug_info=None):
    """Improved green channel interpolation using refined GBTF method."""
    R, G, B = new_img[:, :, 0], new_img[:, :, 1], new_img[:, :, 2]
    S = R + G + B
    
    debug_save(debug_info, S, "1.0_S_sum_of_channels")
    
    # GBTF Step 1: Interpolate using GBTF kernel
    H_interpolated = convolve1d(S, KERNEL_1D_HV, axis=1, mode='reflect')
    V_interpolated = convolve1d(S, KERNEL_1D_HV, axis=0, mode='reflect')
    
    # Create temporary channels based on interpolation
    G_H, R_H, B_H = G.copy(), R.copy(), B.copy()
    G_V, R_V, B_V = G.copy(), R.copy(), B.copy()
    
    # Update specific locations based on GBTF pattern
    G_H[0::2, 0::2] = H_interpolated[0::2, 0::2]  # G at R locations
    G_H[1::2, 1::2] = H_interpolated[1::2, 1::2]  # G at B locations
    G_V[0::2, 0::2] = V_interpolated[0::2, 0::2]
    G_V[1::2, 1::2] = V_interpolated[1::2, 1::2]
    
    # R/B updates for delta computation
    R_H[0::2, 1::2] = H_interpolated[0::2, 1::2]
    B_H[1::2, 0::2] = H_interpolated[1::2, 0::2]
    R_V[1::2, 0::2] = V_interpolated[1::2, 0::2]
    B_V[0::2, 1::2] = V_interpolated[0::2, 1::2]
    
    # GBTF Step 2: Compute deltas
    delta_H = G_H - R_H - B_H
    delta_V = G_V - R_V - B_V
    
    debug_save(debug_info, delta_H, "1.1_delta_H_GBTF")
    debug_save(debug_info, delta_V, "1.1_delta_V_GBTF")
    
    # GBTF Step 3: Gradient calculation with slight smoothing for robustness
    GRAD_KERNEL_1D = np.array([-1, 0, 1], dtype=np.float32)
    
    # Apply minimal smoothing to delta before gradient to reduce noise
    smooth_kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    delta_H_smooth = convolve1d(delta_H, smooth_kernel, axis=1, mode='reflect')
    delta_V_smooth = convolve1d(delta_V, smooth_kernel, axis=0, mode='reflect')
    
    D_H = np.abs(convolve1d(delta_H_smooth, GRAD_KERNEL_1D, axis=1, mode='reflect'))
    D_V = np.abs(convolve1d(delta_V_smooth, GRAD_KERNEL_1D, axis=0, mode='reflect'))
    
    debug_save(debug_info, D_H, "1.2_gradient_H")
    debug_save(debug_info, D_V, "1.2_gradient_V")
    
    # GBTF Step 4: Directional weights with slight modification
    weight_kernels = create_gbtf_weight_kernels()
    W_W, W_E, W_N, W_S, W_T = process_gbtf_weights(D_H, D_V, weight_kernels)
    
    # GBTF Step 5: Final delta computation
    weight_data = (W_N, W_S, W_E, W_W, W_T)
    
    # Process for c=0 pattern (G-R estimation)
    delta_GR = compute_gbtf_delta((delta_H, delta_V), (0,), weight_data)
    
    # Process for c=1 pattern (G-B estimation)  
    delta_GB = compute_gbtf_delta((delta_H, delta_V), (1,), weight_data)
    
    # Apply minimal post-processing to reduce checkerboard in very flat regions
    # Detect very smooth regions
    local_variance = convolve2d(np.abs(S - np.mean(S)), np.ones((3,3))/9, mode='same')
    very_smooth = local_variance < 0.5  # Very conservative threshold
    
    if np.any(very_smooth):
        # Apply mild smoothing only to delta values in very smooth regions
        smooth_kernel_2d = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
        delta_GR_smooth = convolve2d(delta_GR, smooth_kernel_2d, mode='same')
        delta_GB_smooth = convolve2d(delta_GB, smooth_kernel_2d, mode='same')
        
        # Blend only in very smooth regions
        alpha = 0.3  # Conservative blend factor
        delta_GR[very_smooth] = (1-alpha) * delta_GR[very_smooth] + alpha * delta_GR_smooth[very_smooth]
        delta_GB[very_smooth] = (1-alpha) * delta_GB[very_smooth] + alpha * delta_GB_smooth[very_smooth]
    
    debug_save(debug_info, delta_GR, "1.3_delta_GR")
    debug_save(debug_info, delta_GB, "1.3_delta_GB")
    
    # GBTF Step 6: Recover G channel
    new_img[0::2, 0::2, 1] = new_img[0::2, 0::2, 0] + delta_GR[0::2, 0::2]
    new_img[1::2, 1::2, 1] = new_img[1::2, 1::2, 2] + delta_GB[1::2, 1::2]
    
    debug_save(debug_info, new_img[:, :, 1], "1.6_Final_G_Channel_Interpolated")
    
    return new_img


def interpolate_rb_channels_original(new_img, R_mask, G_mask_combined, B_mask, debug_info=None):
    """Interpolate R and B channels using the original fixed-kernel DLMMSE method."""
    G_interpolated = new_img[:, :, 1].copy()
    
    diff_GR = G_interpolated - new_img[:, :, 0]
    diff_GB = G_interpolated - new_img[:, :, 2]
    
    debug_save(debug_info, diff_GR, "2.1_orig_diff_GR_initial_full")
    debug_save(debug_info, diff_GB, "2.1_orig_diff_GB_initial_full")

    delta_GR_diag = convolve2d(diff_GR, KERNEL_DIAGONAL, mode='same')
    delta_GB_diag = convolve2d(diff_GB, KERNEL_DIAGONAL, mode='same')

    debug_save(debug_info, delta_GR_diag, "2.1_orig_delta_GR_interpolated_diag")
    debug_save(debug_info, delta_GB_diag, "2.1_orig_delta_GB_interpolated_diag")
    
    new_img[B_mask, 0] = G_interpolated[B_mask] - delta_GR_diag[B_mask]
    new_img[R_mask, 2] = G_interpolated[R_mask] - delta_GB_diag[R_mask]
    
    debug_save(debug_info, new_img[:, :, 0], "2.1_orig_R_after_B_site_interpolation")
    debug_save(debug_info, new_img[:, :, 2], "2.1_orig_B_after_R_site_interpolation")

    diff_GR_partial = G_interpolated - new_img[:, :, 0]
    diff_GB_partial = G_interpolated - new_img[:, :, 2]

    debug_save(debug_info, diff_GR_partial, "2.2_orig_diff_GR_partial")
    debug_save(debug_info, diff_GB_partial, "2.2_orig_diff_GB_partial")

    delta_GR_cross = convolve2d(diff_GR_partial, KERNEL_CROSS, mode='same')
    delta_GB_cross = convolve2d(diff_GB_partial, KERNEL_CROSS, mode='same')

    debug_save(debug_info, delta_GR_cross, "2.2_orig_delta_GR_interpolated_cross")
    debug_save(debug_info, delta_GB_cross, "2.2_orig_delta_GB_interpolated_cross")
        
    new_img[G_mask_combined, 0] = G_interpolated[G_mask_combined] - delta_GR_cross[G_mask_combined]
    new_img[G_mask_combined, 2] = G_interpolated[G_mask_combined] - delta_GB_cross[G_mask_combined]
    
    debug_save(debug_info, new_img[:, :, 0], "2.3_orig_Final_R_Channel")
    debug_save(debug_info, new_img[:, :, 2], "2.3_orig_Final_B_Channel")
        
    return new_img


def compute_adaptive_diagonal_interpolation(diff_channel, G_interpolated, fixed_kernel_result):
    """Compute adaptive diagonal interpolation with gradient-based weights."""
    G_pad = np.pad(G_interpolated, ((1, 1), (1, 1)), mode='reflect')
    
    # Compute gradients for adaptive weights
    grad_diag1 = np.abs(G_pad[:-2, :-2] - G_pad[2:, 2:]) + EPSILON_RB_ADAPT
    grad_diag2 = np.abs(G_pad[:-2, 2:] - G_pad[2:, :-2]) + EPSILON_RB_ADAPT
    
    # Use smoother weight function
    weight_diag1 = 1.0 / (1.0 + grad_diag1)
    weight_diag2 = 1.0 / (1.0 + grad_diag2)
    
    # Normalize weights
    sum_weights_diag = weight_diag1 + weight_diag2
    weight_diag1 /= sum_weights_diag
    weight_diag2 /= sum_weights_diag

    # Apply convolution-like operation
    diff_padded = np.pad(diff_channel, ((1, 1), (1, 1)), mode='reflect')
    
    # Diagonal averaging
    interp_diag1 = (diff_padded[:-2, :-2] + diff_padded[2:, 2:]) / 2.0
    interp_diag2 = (diff_padded[:-2, 2:] + diff_padded[2:, :-2]) / 2.0
    
    # Weighted combination
    interpolated = interp_diag1 * weight_diag1 + interp_diag2 * weight_diag2
    
    # Fallback to fixed kernel in low-gradient areas
    gradient_strength = (grad_diag1 + grad_diag2) / 2.0
    alpha = np.clip(gradient_strength / 10.0, 0, 1)  # Blend factor
    interpolated = (1 - alpha) * fixed_kernel_result + alpha * interpolated
    
    return interpolated


def interpolate_rb_channels_enhanced(new_img, R_mask, G_mask_01, G_mask_10, B_mask, debug_info=None):
    """Enhanced edge-aware RB channel interpolation with improved quality."""
    G_interpolated = new_img[:, :, 1].copy()
    
    # Initialize difference arrays
    diff_GR = G_interpolated - new_img[:, :, 0]
    diff_GB = G_interpolated - new_img[:, :, 2]

    debug_save(debug_info, diff_GR, "2.1_enh_diff_GR_initial_full")
    debug_save(debug_info, diff_GB, "2.1_enh_diff_GB_initial_full")

    # Diagonal interpolation for R channel at B sites
    delta_GR_fixed = convolve2d(diff_GR, KERNEL_DIAGONAL, mode='same')
    interpolated_dGR = compute_adaptive_diagonal_interpolation(diff_GR, G_interpolated, delta_GR_fixed)
    
    debug_save(debug_info, interpolated_dGR, "2.2_enh_interpolated_dGR_for_B_sites")
    new_img[B_mask, 0] = G_interpolated[B_mask] - interpolated_dGR[B_mask]
    debug_save(debug_info, new_img[:, :, 0], "2.3_enh_R_channel_after_B_site_interpolation")

    # Diagonal interpolation for B channel at R sites
    delta_GB_fixed = convolve2d(diff_GB, KERNEL_DIAGONAL, mode='same')
    interpolated_dGB = compute_adaptive_diagonal_interpolation(diff_GB, G_interpolated, delta_GB_fixed)

    debug_save(debug_info, interpolated_dGB, "2.4_enh_interpolated_dGB_for_R_sites")
    new_img[R_mask, 2] = G_interpolated[R_mask] - interpolated_dGB[R_mask]
    debug_save(debug_info, new_img[:, :, 2], "2.5_enh_B_channel_after_R_site_interpolation")

    # Cross interpolation at G sites
    diff_GR_partial = G_interpolated - new_img[:, :, 0]
    diff_GB_partial = G_interpolated - new_img[:, :, 2]
    
    debug_save(debug_info, diff_GR_partial, "2.6_enh_diff_GR_partial_after_diag_interp")
    debug_save(debug_info, diff_GB_partial, "2.6_enh_diff_GB_partial_after_diag_interp")

    # Use fixed kernel as base
    delta_GR_cross_fixed = convolve2d(diff_GR_partial, KERNEL_CROSS, mode='same')
    delta_GB_cross_fixed = convolve2d(diff_GB_partial, KERNEL_CROSS, mode='same')
    
    # Apply to G mask locations
    G_mask_combined = G_mask_01 | G_mask_10
    
    new_img[G_mask_combined, 0] = G_interpolated[G_mask_combined] - delta_GR_cross_fixed[G_mask_combined]
    new_img[G_mask_combined, 2] = G_interpolated[G_mask_combined] - delta_GB_cross_fixed[G_mask_combined]
    
    debug_save(debug_info, new_img[:, :, 0], "2.8_enh_Final_R_Channel")
    debug_save(debug_info, new_img[:, :, 2], "2.8_enh_Final_B_Channel")
        
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
    
    debug_save(debug_info_tuple, new_img_float, "0.0_input_bayer_img")
    
    # Create Bayer masks
    R_mask, G_mask_combined, G_mask_01, G_mask_10, B_mask = create_bayer_masks(height, width, pattern)
    
    if debug_mode and debug_info_tuple:
        save_dbg_func = debug_info_tuple[1]
        save_dbg_func(R_mask, "0.1_R_mask", is_mask_local=True)
        save_dbg_func(G_mask_01, "0.1_G_mask_01_type1", is_mask_local=True)
        save_dbg_func(G_mask_10, "0.1_G_mask_10_type2", is_mask_local=True)
        save_dbg_func(G_mask_combined, "0.1_G_mask_combined", is_mask_local=True)
        save_dbg_func(new_img_float[:, :, 0] * R_mask, "0.2_initial_R_channel_masked")
        save_dbg_func(new_img_float[:, :, 1] * G_mask_combined, "0.2_initial_G_channel_masked")
        save_dbg_func(new_img_float[:, :, 2] * B_mask, "0.2_initial_B_channel_masked")
    
    # Interpolate green channel
    new_img_float = interpolate_green_channel(new_img_float, R_mask, G_mask_01, G_mask_10, B_mask, debug_info_tuple)
    
    # Interpolate RB channels
    if enable_edge_aware_rb_interpolation:
        if debug_mode: 
            print("Using Enhanced Edge-Aware RB Interpolation (Optimized).")
        new_img_float = interpolate_rb_channels_enhanced(new_img_float, R_mask, G_mask_01, G_mask_10, B_mask, debug_info_tuple)
    else:
        if debug_mode: 
            print("Using Original Fixed-Kernel RB Interpolation.")
        new_img_float = interpolate_rb_channels_original(new_img_float, R_mask, G_mask_combined, B_mask, debug_info_tuple)
    
    # Final conversion
    final_demosaiced_img = np.clip(new_img_float + 0.5, 0, 255).astype(np.uint8)
    
    if debug_mode and debug_info_tuple:
        save_dbg_func = debug_info_tuple[1]
        rb_mode_suffix = "_RBenhanced" if enable_edge_aware_rb_interpolation else "_RBoriginal"
        
        features_str = ""
        if enable_adaptive_threshold:
            features_str += "_AdaptThresh"
            
        save_dbg_func(final_demosaiced_img, f"3.0_Final_Demosaiced_Image{rb_mode_suffix}{features_str}")
    
    return final_demosaiced_img


def load_test_image(input_image_path):
    """Load test image from path with fallback."""
    try:
        img_bgr_uint8 = cv2.imread(input_image_path)
        if img_bgr_uint8 is None:
            # Try alternative path
            alt_input_path = os.path.join("..", os.path.dirname(input_image_path), 
                                         os.path.basename(input_image_path))
            img_bgr_uint8 = cv2.imread(alt_input_path)
            if img_bgr_uint8 is None:
                raise FileNotFoundError(f"Image file not found at {input_image_path} or {alt_input_path}")
            input_image_path = alt_input_path
        return cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB), input_image_path
    except Exception as e:
        print(f"Error loading image: {e}")
        raise


def get_test_configuration_string():
    """Get configuration string for test output naming."""
    features_str = ""
    if enable_adaptive_threshold:
        features_str += "AdaptThresh_"
    
    rb_mode_str = "EdgeAwareRB" if enable_edge_aware_rb_interpolation else "OriginalRB"
    return features_str, rb_mode_str


def test():
    """Runs a self-contained test of the DLMMSE demosaicing algorithm."""
    print(f"Starting self-test for DLMMSE (Optimized Version)...")
    print(f"Edge-Aware RB Interpolation ENABLED: {enable_edge_aware_rb_interpolation}")
    
    print(f"Novel features enabled:")
    print(f"   - Adaptive Threshold: {enable_adaptive_threshold}")
    
    global FileName_DLMMSE
    original_global_filename_prefix = FileName_DLMMSE
    
    features_str, rb_mode_str = get_test_configuration_string()
    FileName_DLMMSE = f"UnitTest_DLMMSE_{features_str}{rb_mode_str}_Optimized"

    # Configure test paths
    input_image_filename = "RGGB_0.0_input_bayer_img.png"
    base_data_dir = "Data"
    input_subdir = "DLMMSE1"
    input_image_path = os.path.join(base_data_dir, input_subdir, input_image_filename)
    test_pattern = 'RGGB'
    
    print(f"Attempting to load Bayer image from: {input_image_path}")
    
    try:
        img_rgb_uint8, actual_path = load_test_image(input_image_path)
        input_image_path = actual_path
    except Exception as e:
        print(f"ERROR: {e}")
        FileName_DLMMSE = original_global_filename_prefix
        return
    
    # Set up output directory
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
        img_rgb_uint8, _ = load_test_image(input_image_path)
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