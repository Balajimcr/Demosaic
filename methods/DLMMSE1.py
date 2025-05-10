import numpy as np
from scipy.signal import convolve2d # Needed for original RB interpolation
from scipy.ndimage import convolve1d # Used in Green channel interpolation
import os
import imageio
import cv2

from methods.Utils import create_bayer_masks, update_filename, save_image, load_image

# Global configurations
debug_mode = True
FileName_DLMMSE = "DLMMSE_Hybrid"
enable_adaptive_directional_G = True # For Green channel adaptive interpolation
enable_hybrid_green = True # New flag for Green channel hybrid enhancement
enable_edge_aware_rb_interpolation = False # True for enhanced RB, False for original RB

# Pre-defined kernels
KERNEL_1D_HV = np.array([-1, 2, 2, 2, -1]) / 4.0
GAUSSIAN_FILTER_1D = np.array([4, 9, 15, 23, 26, 23, 15, 9, 4]) / 128.0
MEAN_FILTER_1D = np.ones(9) / 9.0
KERNEL_DIAGONAL = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]) / 4.0
KERNEL_CROSS = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4.0

# Core Algorithm Functions
def interpolate_green_channel(new_img, R_mask, G_mask_01, G_mask_10, B_mask, debug_info=None):
    """Interpolate the G channel at R and B positions with optional hybrid enhancement."""
    R, G, B = new_img[:, :, 0], new_img[:, :, 1], new_img[:, :, 2]
    S = R + G + B
    
    if debug_info:
        pattern, save = debug_info
        save(S, "1.0_S_sum_of_channels")
    
    # Compute fixed interpolation
    fixed_kernel_g = np.array([1/2, 0, 1/2])
    H_fixed = convolve1d(S, fixed_kernel_g, axis=1, mode='reflect')
    V_fixed = convolve1d(S, fixed_kernel_g, axis=0, mode='reflect')

    if enable_adaptive_directional_G:
        # Compute adaptive interpolation
        epsilon_g_adapt = 1e-5
        padded_S_horiz = np.pad(S, ((0, 0), (1, 1)), mode='reflect')
        left = padded_S_horiz[:, :-2]
        right = padded_S_horiz[:, 2:]
        weight_left = 1.0 / (np.abs(S - left) + epsilon_g_adapt)
        weight_right = 1.0 / (np.abs(S - right) + epsilon_g_adapt)
        sum_weights_h = weight_left + weight_right
        sum_weights_h = np.where(sum_weights_h == 0, 1, sum_weights_h)
        H_adapt = (left * weight_left + right * weight_right) / sum_weights_h
        
        padded_S_vert = np.pad(S, ((1, 1), (0, 0)), mode='reflect')
        up = padded_S_vert[:-2, :]
        down = padded_S_vert[2:, :]
        weight_up = 1.0 / (np.abs(S - up) + epsilon_g_adapt)
        weight_down = 1.0 / (np.abs(S - down) + epsilon_g_adapt)
        sum_weights_v = weight_up + weight_down
        sum_weights_v = np.where(sum_weights_v == 0, 1, sum_weights_v)
        V_adapt = (up * weight_up + down * weight_down) / sum_weights_v
        
        if enable_hybrid_green:
            # Hybrid enhancement: blend fixed and adaptive based on gradient magnitude
            T = 30.0 # Threshold for blending
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
            
            if debug_info:
                save(H, "1.1_H_adaptive_interpolation")
                save(V, "1.1_V_adaptive_interpolation")
    else:
        H = H_fixed
        V = V_fixed
        
        if debug_info:
            save(H, "1.1_H_fixed_interpolation")
            save(V, "1.1_V_fixed_interpolation")
    
    delta_H = H - S
    delta_V = V - S
    
    if debug_info:
        save(delta_H, "1.1_delta_H_before_sign_flip")
        save(delta_V, "1.1_delta_V_before_sign_flip")
    
    delta_H[G_mask_01] = -delta_H[G_mask_01]
    delta_H[G_mask_10] = -delta_H[G_mask_10]
    delta_V[G_mask_01] = -delta_V[G_mask_01]
    delta_V[G_mask_10] = -delta_V[G_mask_10]
    
    if debug_info:
        save(delta_H, "1.1_delta_H_after_sign_flip")
        save(delta_V, "1.1_delta_V_after_sign_flip")
    
    gaussian_filter_kernel = np.array([1/4, 1/2, 1/4])
    gaussian_H = convolve1d(delta_H, gaussian_filter_kernel, axis=1, mode='reflect')
    gaussian_V = convolve1d(delta_V, gaussian_filter_kernel, axis=0, mode='reflect')
    
    if debug_info:
        save(gaussian_H, "1.2_gaussian_smoothed_delta_H")
        save(gaussian_V, "1.2_gaussian_smoothed_delta_V")
    
    mean_filter_kernel_g = np.array([1/3, 1/3, 1/3])
    mean_H = convolve1d(gaussian_H, mean_filter_kernel_g, axis=1, mode='reflect')
    mean_V = convolve1d(gaussian_V, mean_filter_kernel_g, axis=0, mode='reflect')
    
    epsilon_var = 1e-10
    var_value_H = convolve1d(np.square(gaussian_H - mean_H), mean_filter_kernel_g, axis=1, mode='reflect') + epsilon_var
    var_value_V = convolve1d(np.square(gaussian_V - mean_V), mean_filter_kernel_g, axis=0, mode='reflect') + epsilon_var
    
    var_noise_H = convolve1d(np.square(delta_H - gaussian_H), mean_filter_kernel_g, axis=1, mode='reflect') + epsilon_var
    var_noise_V = convolve1d(np.square(delta_V - gaussian_V), mean_filter_kernel_g, axis=0, mode='reflect') + epsilon_var
    
    if debug_info:
        save(mean_H, "1.3_mean_H")
        save(mean_V, "1.3_mean_V")
        save(var_value_H, "1.3_var_value_H")
        save(var_value_V, "1.3_var_value_V")
        save(var_noise_H, "1.3_var_noise_H")
        save(var_noise_V, "1.3_var_noise_V")
    
    signal_ratio_H = var_value_H / (var_noise_H + var_value_H)
    signal_ratio_V = var_value_V / (var_noise_V + var_value_V)
    
    new_H = mean_H + signal_ratio_H * (delta_H - mean_H)
    new_V = mean_V + signal_ratio_V * (delta_V - mean_V)
    
    if debug_info:
        save(new_H, "1.4_refined_delta_H")
        save(new_V, "1.4_refined_delta_V")
    
    var_x_H = np.abs(var_value_H - var_value_H**2 / (var_value_H + var_noise_H)) + epsilon_var
    var_x_V = np.abs(var_value_V - var_value_V**2 / (var_value_V + var_noise_V)) + epsilon_var
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
    """Interpolate R and B channels using the enhanced edge-aware algorithm."""
    G_interpolated = new_img[:, :, 1].copy()
    R_original_sparse = new_img[:, :, 0].copy()
    B_original_sparse = new_img[:, :, 2].copy()
    epsilon_rb_adapt = 1e-5

    if debug_info:
        pattern, save = debug_info
    
    diff_GR = np.zeros_like(G_interpolated)
    diff_GB = np.zeros_like(G_interpolated)
    diff_GR[R_mask] = G_interpolated[R_mask] - R_original_sparse[R_mask]
    diff_GB[B_mask] = G_interpolated[B_mask] - B_original_sparse[B_mask]

    if debug_info:
        save(diff_GR * R_mask, "2.1_enh_diff_GR_initial_at_R_locations")
        save(diff_GB * B_mask, "2.1_enh_diff_GB_initial_at_B_locations")

    G_pad_diag = np.pad(G_interpolated, ((1,1),(1,1)), mode='reflect')
    grad_G_diag1 = np.abs(G_pad_diag[:-2, :-2] - G_pad_diag[2:, 2:]) + epsilon_rb_adapt
    grad_G_diag2 = np.abs(G_pad_diag[:-2, 2:] - G_pad_diag[2:, :-2]) + epsilon_rb_adapt
    weight_diag1 = 1.0 / grad_G_diag1
    weight_diag2 = 1.0 / grad_G_diag2
    sum_weights_diag = weight_diag1 + weight_diag2

    diff_GR_padded = np.pad(diff_GR, ((1,1),(1,1)), mode='reflect')
    R_mask_padded = np.pad(R_mask.astype(float), ((1,1),(1,1)), mode='reflect')
    dGR_vals_diag1 = diff_GR_padded[:-2, :-2] * R_mask_padded[:-2,:-2] + diff_GR_padded[2:, 2:] * R_mask_padded[2:,2:]
    dGR_counts_diag1 = R_mask_padded[:-2, :-2] + R_mask_padded[2:, 2:]
    dGR_avg_diag1 = np.divide(dGR_vals_diag1, dGR_counts_diag1, out=np.zeros_like(dGR_vals_diag1), where=dGR_counts_diag1!=0)
    dGR_vals_diag2 = diff_GR_padded[:-2, 2:] * R_mask_padded[:-2,2:] + diff_GR_padded[2:, :-2] * R_mask_padded[2:,:-2]
    dGR_counts_diag2 = R_mask_padded[:-2, 2:] + R_mask_padded[2:, :-2]
    dGR_avg_diag2 = np.divide(dGR_vals_diag2, dGR_counts_diag2, out=np.zeros_like(dGR_vals_diag2), where=dGR_counts_diag2!=0)
    interpolated_dGR_for_B_sites = (dGR_avg_diag1 * weight_diag1 + dGR_avg_diag2 * weight_diag2) / sum_weights_diag
    
    if debug_info: save(interpolated_dGR_for_B_sites, "2.2_enh_interpolated_dGR_for_B_sites")
    new_img[B_mask, 0] = G_interpolated[B_mask] - interpolated_dGR_for_B_sites[B_mask]
    if debug_info: save(new_img[:,:,0], "2.3_enh_R_channel_after_B_site_interpolation")

    diff_GB_padded = np.pad(diff_GB, ((1,1),(1,1)), mode='reflect')
    B_mask_padded = np.pad(B_mask.astype(float), ((1,1),(1,1)), mode='reflect')
    dGB_vals_diag1 = diff_GB_padded[:-2, :-2] * B_mask_padded[:-2,:-2] + diff_GB_padded[2:, 2:] * B_mask_padded[2:,2:]
    dGB_counts_diag1 = B_mask_padded[:-2, :-2] + B_mask_padded[2:, 2:]
    dGB_avg_diag1 = np.divide(dGB_vals_diag1, dGB_counts_diag1, out=np.zeros_like(dGB_vals_diag1), where=dGB_counts_diag1!=0)
    dGB_vals_diag2 = diff_GB_padded[:-2, 2:] * B_mask_padded[:-2,2:] + diff_GB_padded[2:, :-2] * B_mask_padded[2:,:-2]
    dGB_counts_diag2 = B_mask_padded[:-2, 2:] + B_mask_padded[2:, :-2]
    dGB_avg_diag2 = np.divide(dGB_vals_diag2, dGB_counts_diag2, out=np.zeros_like(dGB_vals_diag2), where=dGB_counts_diag2!=0)
    interpolated_dGB_for_R_sites = (dGB_avg_diag1 * weight_diag1 + dGB_avg_diag2 * weight_diag2) / sum_weights_diag

    if debug_info: save(interpolated_dGB_for_R_sites, "2.4_enh_interpolated_dGB_for_R_sites")
    new_img[R_mask, 2] = G_interpolated[R_mask] - interpolated_dGB_for_R_sites[R_mask]
    if debug_info: save(new_img[:,:,2], "2.5_enh_B_channel_after_R_site_interpolation")

    diff_GR_partial = G_interpolated - new_img[:, :, 0]
    diff_GB_partial = G_interpolated - new_img[:, :, 2]
    if debug_info:
        save(diff_GR_partial, "2.6_enh_diff_GR_partial_after_diag_interp")
        save(diff_GB_partial, "2.6_enh_diff_GB_partial_after_diag_interp")

    G_padded_H = np.pad(G_interpolated, ((0,0),(1,1)), mode='reflect')
    G_left = G_padded_H[:, :-2]; G_right = G_padded_H[:, 2:]
    grad_G_H_left = np.abs(G_interpolated - G_left) + epsilon_rb_adapt
    grad_G_H_right = np.abs(G_interpolated - G_right) + epsilon_rb_adapt
    weight_G_left = 1.0 / grad_G_H_left; weight_G_right = 1.0 / grad_G_H_right
    sum_weights_G_H = weight_G_left + weight_G_right

    dGRp_padded_H = np.pad(diff_GR_partial, ((0,0),(1,1)), mode='reflect')
    dGRp_left = dGRp_padded_H[:, :-2]; dGRp_right = dGRp_padded_H[:, 2:]
    interpolated_dGRp_H = (dGRp_left * weight_G_left + dGRp_right * weight_G_right) / sum_weights_G_H
    dGBp_padded_H = np.pad(diff_GB_partial, ((0,0),(1,1)), mode='reflect')
    dGBp_left = dGBp_padded_H[:, :-2]; dGBp_right = dGBp_padded_H[:, 2:]
    interpolated_dGBp_H = (dGBp_left * weight_G_left + dGBp_right * weight_G_right) / sum_weights_G_H

    G_padded_V = np.pad(G_interpolated, ((1,1),(0,0)), mode='reflect')
    G_up = G_padded_V[:-2, :]; G_down = G_padded_V[2:, :]
    grad_G_V_up = np.abs(G_interpolated - G_up) + epsilon_rb_adapt
    grad_G_V_down = np.abs(G_interpolated - G_down) + epsilon_rb_adapt
    weight_G_up = 1.0 / grad_G_V_up; weight_G_down = 1.0 / grad_G_V_down
    sum_weights_G_V = weight_G_up + weight_G_down

    dGRp_padded_V = np.pad(diff_GR_partial, ((1,1),(0,0)), mode='reflect')
    dGRp_up = dGRp_padded_V[:-2, :]; dGRp_down = dGRp_padded_V[2:, :]
    interpolated_dGRp_V = (dGRp_up * weight_G_up + dGRp_down * weight_G_down) / sum_weights_G_V
    dGBp_padded_V = np.pad(diff_GB_partial, ((1,1),(0,0)), mode='reflect')
    dGBp_up = dGBp_padded_V[:-2, :]; dGBp_down = dGBp_padded_V[2:, :]
    interpolated_dGBp_V = (dGBp_up * weight_G_up + dGBp_down * weight_G_down) / sum_weights_G_V
    
    if debug_info:
        save(interpolated_dGRp_H, "2.7_enh_interpolated_dGRp_Horizontal")
        save(interpolated_dGRp_V, "2.7_enh_interpolated_dGRp_Vertical")
        save(interpolated_dGBp_H, "2.7_enh_interpolated_dGBp_Horizontal")
        save(interpolated_dGBp_V, "2.7_enh_interpolated_dGBp_Vertical")

    G_mask_combined = G_mask_01 | G_mask_10
    final_interpolated_dGRp_at_G = np.zeros_like(G_interpolated)
    final_interpolated_dGRp_at_G[G_mask_01] = interpolated_dGRp_H[G_mask_01]
    final_interpolated_dGRp_at_G[G_mask_10] = interpolated_dGRp_V[G_mask_10]
    new_img[G_mask_combined, 0] = G_interpolated[G_mask_combined] - final_interpolated_dGRp_at_G[G_mask_combined]

    final_interpolated_dGBp_at_G = np.zeros_like(G_interpolated)
    final_interpolated_dGBp_at_G[G_mask_01] = interpolated_dGBp_V[G_mask_01]
    final_interpolated_dGBp_at_G[G_mask_10] = interpolated_dGBp_H[G_mask_10]
    new_img[G_mask_combined, 2] = G_interpolated[G_mask_combined] - final_interpolated_dGBp_at_G[G_mask_combined]
    
    if debug_info:
        save(new_img[:, :, 0], "2.8_enh_Final_R_Channel")
        save(new_img[:, :, 2], "2.8_enh_Final_B_Channel")
        
    return new_img

def run(img, pattern='RGGB'):
    """DLMMSE demosaicing function with selectable RB interpolation."""
    height, width, _ = img.shape
    
    def save_debug_image_local(image_data, stage_name_local, is_mask_local=False):
        update_filename(pattern)
        save_image(image_data, stage_name_local, pattern=pattern.upper(), is_mask=is_mask_local)
    
    debug_info_tuple = (pattern, save_debug_image_local) if debug_mode else None
    
    new_img_float = np.copy(img).astype(float)
    
    if debug_mode and debug_info_tuple:
        _, save_dbg_func = debug_info_tuple
        save_dbg_func(new_img_float, "0.0_input_bayer_img_float")
    
    R_mask, G_mask_combined, G_mask_01, G_mask_10, B_mask = create_bayer_masks(height, width, pattern)
    
    if debug_mode and debug_info_tuple:
        _, save_dbg_func = debug_info_tuple
        save_dbg_func(R_mask, "0.1_R_mask", is_mask_local=True)
        save_dbg_func(G_mask_01, "0.1_G_mask_01_type1", is_mask_local=True)
        save_dbg_func(G_mask_10, "0.1_G_mask_10_type2", is_mask_local=True)
        save_dbg_func(G_mask_combined, "0.1_G_mask_combined", is_mask_local=True)
        save_dbg_func(B_mask, "0.1_B_mask", is_mask_local=True)
        save_dbg_func(new_img_float[:, :, 0] * R_mask, "0.2_initial_R_channel_masked")
        save_dbg_func(new_img_float[:, :, 1] * G_mask_combined, "0.2_initial_G_channel_masked")
        save_dbg_func(new_img_float[:, :, 2] * B_mask, "0.2_initial_B_channel_masked")
    
    new_img_float = interpolate_green_channel(new_img_float, R_mask, G_mask_01, G_mask_10, B_mask, debug_info_tuple)
    
    if enable_edge_aware_rb_interpolation:
        if debug_mode: print("Using Enhanced Edge-Aware RB Interpolation.")
        new_img_float = interpolate_rb_channels_enhanced(new_img_float, R_mask, G_mask_01, G_mask_10, B_mask, debug_info_tuple)
    else:
        if debug_mode: print("Using Original Fixed-Kernel RB Interpolation.")
        new_img_float = interpolate_rb_channels_original(new_img_float, R_mask, G_mask_combined, B_mask, debug_info_tuple)
    
    final_demosaiced_img = (new_img_float + 0.5).clip(0, 255).astype(np.uint8)
    
    if debug_mode and debug_info_tuple:
        _, save_dbg_func = debug_info_tuple
        rb_mode_suffix = "_RBenhanced" if enable_edge_aware_rb_interpolation else "_RBoriginal"
        save_dbg_func(final_demosaiced_img, f"3.0_Final_Demosaiced_Image{rb_mode_suffix}")
    
    return final_demosaiced_img

# Test Function
def test():
    """Runs a self-contained test of the DLMMSE demosaicing algorithm."""
    print(f"Starting self-test for DLMMSE...")
    print(f"Edge-Aware RB Interpolation ENABLED: {enable_edge_aware_rb_interpolation}")
    
    global FileName_DLMMSE
    original_global_filename_prefix = FileName_DLMMSE
    rb_mode_str = "EdgeAwareRB" if enable_edge_aware_rb_interpolation else "OriginalRB"
    FileName_DLMMSE = f"UnitTest_DLMMSE_{rb_mode_str}"

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
    
    test_output_main_dir = os.path.join(base_data_dir, f"UnitTest_DLMMSE_{rb_mode_str}_Output")
    os.makedirs(test_output_main_dir, exist_ok=True)
    print(f"Test output (final image) will be saved in: {test_output_main_dir}")
    if debug_mode:
        print(f"Debug images (if enabled) will be in a subfolder like: Data/{FileName_DLMMSE}_{test_pattern}/")

    print(f"Running DLMMSE algorithm (RB mode: {'Enhanced' if enable_edge_aware_rb_interpolation else 'Original'}) on loaded image (pattern: {test_pattern})...")
    
    try:
        demosaiced_img = run(img_rgb_uint8, pattern=test_pattern)
        base_input_fn = os.path.basename(input_image_path)
        demosaiced_output_filename = f"demosaiced_{rb_mode_str}_output_of_{base_input_fn}"
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

if __name__ == "__main__":
    print("--- Testing with Enhanced RB Interpolation ---")
    enable_edge_aware_rb_interpolation = True
    test()
    print("\n--- Testing with Original RB Interpolation ---")
    enable_edge_aware_rb_interpolation = False
    test()