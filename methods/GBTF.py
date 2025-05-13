import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d
import os
import imageio

# Global configurations
debug_mode = True
FileName_GBTF = "GBTF"  # Default filename prefix

# --- Algorithmic Constants ---
# Kernels and parameters are defined globally for clarity and to avoid re-creation on each call.
# Using float types for all kernel elements and relevant constants is crucial for precision.

# For Step 1: Interpolation
INTERP_KERNEL_1D = np.array([-1, 2, 2, 2, -1], dtype=float) / 4.0

# For Step 2: Gradient calculation
GRAD_KERNEL_1D = np.array([-1, 0, 1], dtype=float)

# For Step 3: Coefficient calculation (Directional Weights)
WEIGHT_EPSILON = 1e-10  # Small value to prevent division by zero and handle flat regions

KERNEL_W_W_2D = np.zeros((9, 9), dtype=float) # West Weight Kernel
KERNEL_W_W_2D[2:7, 0:5] = 1.0
KERNEL_W_E_2D = np.zeros((9, 9), dtype=float) # East Weight Kernel
KERNEL_W_E_2D[2:7, 4:9] = 1.0
KERNEL_W_N_2D = np.zeros((9, 9), dtype=float) # North Weight Kernel
KERNEL_W_N_2D[0:5, 2:7] = 1.0
KERNEL_W_S_2D = np.zeros((9, 9), dtype=float) # South Weight Kernel
KERNEL_W_S_2D[4:9, 2:7] = 1.0

# For Step 4: Final delta computation
# Original: f = [1, 1, 1, 1, 1]; f = f / np.sum(f) which results in [0.2, 0.2, 0.2, 0.2, 0.2]
F_COEFFS_1D = np.full(5, 0.2, dtype=float)

KERNEL_F_FORWARD_1D = np.zeros(9, dtype=float)
KERNEL_F_FORWARD_1D[0:5] = F_COEFFS_1D
KERNEL_F_BACKWARD_1D = np.zeros(9, dtype=float)
KERNEL_F_BACKWARD_1D[4:9] = F_COEFFS_1D[::-1]  # Reversed coefficients

# For Step 6: Recover R in B-locations, B in R-locations
KERNEL_PRB_2D = np.zeros((7, 7), dtype=float) # Patterned Recovery Kernel
KERNEL_PRB_2D[0, 2] = KERNEL_PRB_2D[0, 4] = KERNEL_PRB_2D[2, 0] = KERNEL_PRB_2D[2, 6] = -1.0
KERNEL_PRB_2D[6, 2] = KERNEL_PRB_2D[6, 4] = KERNEL_PRB_2D[4, 0] = KERNEL_PRB_2D[4, 6] = -1.0
KERNEL_PRB_2D[2, 2] = KERNEL_PRB_2D[2, 4] = KERNEL_PRB_2D[4, 2] = KERNEL_PRB_2D[4, 4] = 10.0
SUM_KERNEL_PRB_2D = np.sum(KERNEL_PRB_2D)  # Expected: 32.0

# For Step 7: Recover R/B in G-locations
KERNEL_RB_G_2D = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float) # Recovery Kernel for R/B at G sites
SUM_KERNEL_RB_G_2D = np.sum(KERNEL_RB_G_2D)  # Expected: 4.0

def update_filename(pattern):
    """Updates the filename prefix for debug image saves."""
    global FileName_GBTF
    FileName_GBTF = pattern

def save_image(image, stage_name, pattern=None, is_mask=False):
    """Saves an image or mask as a PNG file if debug_mode is enabled."""
    if not debug_mode:
        return

    # Create debug directory if it doesn't exist
    foldername = os.path.join("Data/GBTF/")
    os.makedirs(foldername, exist_ok=True)

    # Prepare the image for saving
    if is_mask:
        image_to_save = (image * 255).astype(np.uint8)
        filename = f"{FileName_GBTF}_{stage_name}"
        if pattern:
            filename = f"{FileName_GBTF}_{pattern}_{stage_name}"
        filename += "_mask.png"
    else:
        image_to_save = image.clip(0, 255).astype(np.uint8)
        filename = f"{FileName_GBTF}_{stage_name}.png"

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
    pattern = pattern.upper()
    
    # Create binary masks
    R_mask = np.zeros((height, width), dtype=bool)
    G_mask_01 = np.zeros((height, width), dtype=bool)
    G_mask_10 = np.zeros((height, width), dtype=bool)
    B_mask = np.zeros((height, width), dtype=bool)
    
    if pattern == 'RGGB':
        R_mask[0::2, 0::2] = True
        G_mask_01[0::2, 1::2] = True
        G_mask_10[1::2, 0::2] = True
        B_mask[1::2, 1::2] = True
    elif pattern == 'BGGR':
        B_mask[0::2, 0::2] = True
        G_mask_01[0::2, 1::2] = True
        G_mask_10[1::2, 0::2] = True
        R_mask[1::2, 1::2] = True
    elif pattern == 'GRBG':
        G_mask_01[0::2, 0::2] = True
        R_mask[0::2, 1::2] = True
        B_mask[1::2, 0::2] = True
        G_mask_10[1::2, 1::2] = True
    elif pattern == 'GBRG':
        G_mask_01[0::2, 0::2] = True
        B_mask[0::2, 1::2] = True
        R_mask[1::2, 0::2] = True
        G_mask_10[1::2, 1::2] = True
    
    # Combined G mask
    G_mask = G_mask_01 | G_mask_10
    
    return R_mask, G_mask, G_mask_01, G_mask_10, B_mask

def run(img: np.ndarray, pattern: str = 'RGGB') -> np.ndarray:
    """
    Processes an image using GBTF algorithm adapted for multiple Bayer patterns.
    This implementation directly handles different patterns without transformation.
    
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
    
    # Validate pattern
    pattern = pattern.upper()
    supported_patterns = ['RGGB', 'BGGR', 'GRBG', 'GBRG']
    if pattern not in supported_patterns:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}. Supported patterns are: {supported_patterns}")
    
    S = img.astype(float)
    R, G, B = S[:, :, 0], S[:, :, 1], S[:, :, 2]

    # Initialize new color channels by copying the original ones
    new_R, new_G, new_B = np.copy(R), np.copy(G), np.copy(B)

    sum_RGB = R + G + B
    
    # Create masks
    R_mask, G_mask, G_mask_01, G_mask_10, B_mask = create_bayer_masks(height, width, pattern)
    
    if debug_mode:
        save_debug_image(S, "0.0_input_bayer_img")
        save_debug_image(sum_RGB, "0.1_sum_RGB")
        save_debug_image(R_mask, "R_mask", is_mask=True)
        save_debug_image(G_mask_01, "G_mask_01", is_mask=True)
        save_debug_image(G_mask_10, "G_mask_10", is_mask=True)
        save_debug_image(B_mask, "B_mask", is_mask=True)

    # Step 1: Interpolate color information horizontally and vertically
    H_interpolated = convolve1d(sum_RGB, INTERP_KERNEL_1D, axis=1, mode='reflect')
    V_interpolated = convolve1d(sum_RGB, INTERP_KERNEL_1D, axis=0, mode='reflect')
    
    if debug_mode:
        save_debug_image(H_interpolated, "1.0_H_interpolated")
        save_debug_image(V_interpolated, "1.0_V_interpolated")

    # Create temporary channel versions based on interpolation
    G_H, R_H, B_H = np.copy(G), np.copy(R), np.copy(B)
    G_V, R_V, B_V = np.copy(G), np.copy(R), np.copy(B)
    
    # The original GBTF logic for RGGB is very specific:
    # G_H gets H at R and B locations
    # R_H gets H at G1 locations (not B locations!)
    # B_H gets H at G2 locations (not R locations!)
    # This creates a cross-pattern interpolation
    
    # Horizontal interpolation
    G_H[R_mask] = H_interpolated[R_mask]  # G at R locations
    G_H[B_mask] = H_interpolated[B_mask]  # G at B locations
    
    # For RGGB: R_H at (0,1), B_H at (1,0) - this is the key insight!
    # We need to identify which G positions get which interpolation
    if pattern == 'RGGB':
        R_H[G_mask_01] = H_interpolated[G_mask_01]  # R at G1
        B_H[G_mask_10] = H_interpolated[G_mask_10]  # B at G2
    elif pattern == 'BGGR':
        B_H[G_mask_01] = H_interpolated[G_mask_01]  # B at G1
        R_H[G_mask_10] = H_interpolated[G_mask_10]  # R at G2
    elif pattern == 'GRBG':
        R_H[G_mask_01] = H_interpolated[G_mask_01]  # R at G1
        B_H[G_mask_10] = H_interpolated[G_mask_10]  # B at G2
    elif pattern == 'GBRG':
        B_H[G_mask_01] = H_interpolated[G_mask_01]  # B at G1
        R_H[G_mask_10] = H_interpolated[G_mask_10]  # R at G2
    
    # Vertical interpolation (opposite of horizontal)
    G_V[R_mask] = V_interpolated[R_mask]  # G at R locations
    G_V[B_mask] = V_interpolated[B_mask]  # G at B locations
    
    # For RGGB: R_V at (1,0), B_V at (0,1) - opposite of horizontal!
    if pattern == 'RGGB':
        R_V[G_mask_10] = V_interpolated[G_mask_10]  # R at G2
        B_V[G_mask_01] = V_interpolated[G_mask_01]  # B at G1
    elif pattern == 'BGGR':
        B_V[G_mask_10] = V_interpolated[G_mask_10]  # B at G2
        R_V[G_mask_01] = V_interpolated[G_mask_01]  # R at G1
    elif pattern == 'GRBG':
        R_V[G_mask_10] = V_interpolated[G_mask_10]  # R at G2
        B_V[G_mask_01] = V_interpolated[G_mask_01]  # B at G1
    elif pattern == 'GBRG':
        B_V[G_mask_10] = V_interpolated[G_mask_10]  # B at G2
        R_V[G_mask_01] = V_interpolated[G_mask_01]  # R at G1
    
    if debug_mode:
        save_debug_image(G_H, "1.1_G_H")
        save_debug_image(R_H, "1.1_R_H")
        save_debug_image(B_H, "1.1_B_H")
        save_debug_image(G_V, "1.1_G_V")
        save_debug_image(R_V, "1.1_R_V")
        save_debug_image(B_V, "1.1_B_V")

    # Step 2: Compute delta values (differences) and their gradients
    delta_H = G_H - R_H - B_H
    delta_V = G_V - R_V - B_V
    
    if debug_mode:
        save_debug_image(delta_H, "2.0_delta_H")
        save_debug_image(delta_V, "2.0_delta_V")

    # Gradient calculation
    D_H = np.absolute(convolve1d(delta_H, GRAD_KERNEL_1D, axis=1, mode='reflect'))
    D_V = np.absolute(convolve1d(delta_V, GRAD_KERNEL_1D, axis=0, mode='reflect'))
    
    if debug_mode:
        save_debug_image(D_H, "2.1_D_H_gradients")
        save_debug_image(D_V, "2.1_D_V_gradients")

    # Step 3: Compute directional weight coefficients
    W_W = convolve2d(D_H, KERNEL_W_W_2D, mode='same', boundary='fill', fillvalue=0.0)
    W_E = convolve2d(D_H, KERNEL_W_E_2D, mode='same', boundary='fill', fillvalue=0.0)
    W_N = convolve2d(D_V, KERNEL_W_N_2D, mode='same', boundary='fill', fillvalue=0.0)
    W_S = convolve2d(D_V, KERNEL_W_S_2D, mode='same', boundary='fill', fillvalue=0.0)
    
    if debug_mode:
        save_debug_image(W_W, "3.0_W_W_before_inversion")
        save_debug_image(W_E, "3.0_W_E_before_inversion")
        save_debug_image(W_N, "3.0_W_N_before_inversion")
        save_debug_image(W_S, "3.0_W_S_before_inversion")

    # Process weights
    W_W[W_W == 0] = WEIGHT_EPSILON; W_W = 1.0 / np.square(W_W)
    W_E[W_E == 0] = WEIGHT_EPSILON; W_E = 1.0 / np.square(W_E)
    W_N[W_N == 0] = WEIGHT_EPSILON; W_N = 1.0 / np.square(W_N)
    W_S[W_S == 0] = WEIGHT_EPSILON; W_S = 1.0 / np.square(W_S)

    W_T = W_W + W_E + W_N + W_S  # Total weight
    
    if debug_mode:
        save_debug_image(W_W, "3.1_W_W_after_inversion")
        save_debug_image(W_E, "3.1_W_E_after_inversion")
        save_debug_image(W_N, "3.1_W_N_after_inversion")
        save_debug_image(W_S, "3.1_W_S_after_inversion")
        save_debug_image(W_T, "3.1_W_T_total")

    # Step 4: Compute final delta values
    # For GR (Green-Red): process rows/cols containing R
    # For GB (Green-Blue): process rows/cols containing B
    
    # Determine which rows and columns contain R and B pixels
    r_rows = np.where(R_mask.any(axis=1))[0][0] % 2
    r_cols = np.where(R_mask.any(axis=0))[0][0] % 2
    b_rows = np.where(B_mask.any(axis=1))[0][0] % 2
    b_cols = np.where(B_mask.any(axis=0))[0][0] % 2
    
    # Process delta_GR
    current_delta_H_GR = np.zeros_like(delta_H)
    current_delta_V_GR = np.zeros_like(delta_V)
    current_delta_H_GR[r_rows::2, :] = delta_H[r_rows::2, :]
    current_delta_V_GR[:, r_cols::2] = delta_V[:, r_cols::2]
    
    if debug_mode:
        save_debug_image(current_delta_H_GR, "4.0_current_delta_H_GR")
        save_debug_image(current_delta_V_GR, "4.0_current_delta_V_GR")
    
    # Convolutions for GR
    V1_N_GR = convolve1d(current_delta_V_GR, KERNEL_F_FORWARD_1D, axis=0, mode='reflect')
    V2_S_GR = convolve1d(current_delta_V_GR, KERNEL_F_BACKWARD_1D, axis=0, mode='reflect')
    V3_E_GR = convolve1d(current_delta_H_GR, KERNEL_F_FORWARD_1D, axis=1, mode='reflect')
    V4_W_GR = convolve1d(current_delta_H_GR, KERNEL_F_BACKWARD_1D, axis=1, mode='reflect')
    
    delta_GR = (V1_N_GR * W_N + V2_S_GR * W_S + V3_E_GR * W_E + V4_W_GR * W_W) / W_T
    
    if debug_mode:
        save_debug_image(delta_GR, "4.0_delta_GR")
    
    # Process delta_GB
    current_delta_H_GB = np.zeros_like(delta_H)
    current_delta_V_GB = np.zeros_like(delta_V)
    current_delta_H_GB[b_rows::2, :] = delta_H[b_rows::2, :]
    current_delta_V_GB[:, b_cols::2] = delta_V[:, b_cols::2]
    
    if debug_mode:
        save_debug_image(current_delta_H_GB, "4.1_current_delta_H_GB")
        save_debug_image(current_delta_V_GB, "4.1_current_delta_V_GB")
    
    # Convolutions for GB
    V1_N_GB = convolve1d(current_delta_V_GB, KERNEL_F_FORWARD_1D, axis=0, mode='reflect')
    V2_S_GB = convolve1d(current_delta_V_GB, KERNEL_F_BACKWARD_1D, axis=0, mode='reflect')
    V3_E_GB = convolve1d(current_delta_H_GB, KERNEL_F_FORWARD_1D, axis=1, mode='reflect')
    V4_W_GB = convolve1d(current_delta_H_GB, KERNEL_F_BACKWARD_1D, axis=1, mode='reflect')
    
    delta_GB = (V1_N_GB * W_N + V2_S_GB * W_S + V3_E_GB * W_E + V4_W_GB * W_W) / W_T
    
    if debug_mode:
        save_debug_image(delta_GB, "4.1_delta_GB")

    # Step 5: Recover G channel
    new_G[R_mask] = R[R_mask] + delta_GR[R_mask]
    new_G[B_mask] = B[B_mask] + delta_GB[B_mask]
    
    if debug_mode:
        save_debug_image(new_G, "5.0_new_G")

    # Step 6: Recover R at B-locations and B at R-locations
    convolved_delta_for_R_at_B = convolve2d(delta_GR, KERNEL_PRB_2D, mode='same', boundary='fill', fillvalue=0.0) / SUM_KERNEL_PRB_2D
    new_R[B_mask] = (new_G - convolved_delta_for_R_at_B)[B_mask]

    convolved_delta_for_B_at_R = convolve2d(delta_GB, KERNEL_PRB_2D, mode='same', boundary='fill', fillvalue=0.0) / SUM_KERNEL_PRB_2D
    new_B[R_mask] = (new_G - convolved_delta_for_B_at_R)[R_mask]
    
    if debug_mode:
        save_debug_image(new_R, "6.0_new_R_after_RB_recovery")
        save_debug_image(new_B, "6.0_new_B_after_RB_recovery")

    # Step 7: Recover R and B at G-locations
    diff_G_newR = new_G - new_R
    R_at_G_correction = convolve2d(diff_G_newR, KERNEL_RB_G_2D, mode='same', boundary='fill', fillvalue=0.0) / SUM_KERNEL_RB_G_2D
    R_interp_at_G_locations = G - R_at_G_correction

    new_R[G_mask] = R_interp_at_G_locations[G_mask]

    diff_G_newB = new_G - new_B
    B_at_G_correction = convolve2d(diff_G_newB, KERNEL_RB_G_2D, mode='same', boundary='fill', fillvalue=0.0) / SUM_KERNEL_RB_G_2D
    B_interp_at_G_locations = G - B_at_G_correction

    new_B[G_mask] = B_interp_at_G_locations[G_mask]
    
    if debug_mode:
        save_debug_image(new_R, "7.0_final_R")
        save_debug_image(new_B, "7.0_final_B")

    # Step 8: Finalize image construction
    final_image_float = np.dstack((new_R, new_G, new_B))
    
    if debug_mode:
        save_debug_image(final_image_float, "8.0_final_image_float")

    # Round by adding 0.5, then clip to [0, 255] and cast to uint8
    final_image_uint8 = np.clip(final_image_float + 0.5, 0, 255).astype(np.uint8)
    
    if debug_mode:
        save_debug_image(final_image_uint8, "8.1_final_image_uint8")

    return final_image_uint8