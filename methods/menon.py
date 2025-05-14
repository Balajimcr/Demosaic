import numpy as np
from scipy.ndimage import convolve, convolve1d
import os
import imageio

# Global configurations
debug_mode = False
FileName_Menon = "Menon"  # Default filename prefix

# --- Algorithmic Constants ---
# Kernels for directional interpolation
H_0_1D = np.array([0.0, 0.5, 0.0, 0.5, 0.0], dtype=float)
H_1_1D = np.array([-0.25, 0.0, 0.5, 0.0, -0.25], dtype=float)

# Kernel for directional decision
KERNEL_DECISION_2D = np.array([
    [0.0, 0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 3.0, 0.0, 3.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 1.0],
], dtype=float)

# Kernel for basic interpolation
K_B_1D = np.array([0.5, 0, 0.5], dtype=float)

# FIR filter for refining step
FIR_1D = np.ones(3, dtype=float) / 3.0

def update_filename(pattern):
    """Updates the filename prefix for debug image saves."""
    global FileName_Menon
    FileName_Menon = pattern

def save_image(image, stage_name, pattern=None, is_mask=False):
    """Saves an image or mask as a PNG file if debug_mode is enabled."""
    if not debug_mode:
        return

    # Create debug directory if it doesn't exist
    foldername = os.path.join("Data/Menon/")
    os.makedirs(foldername, exist_ok=True)

    # Prepare the image for saving
    if is_mask:
        image_to_save = (image * 255).astype(np.uint8)
        filename = f"{FileName_Menon}_{stage_name}"
        if pattern:
            filename = f"{FileName_Menon}_{pattern}_{stage_name}"
        filename += "_mask.png"
    else:
        image_to_save = image.clip(0, 255).astype(np.uint8)
        filename = f"{FileName_Menon}_{stage_name}.png"

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
    G_mask = np.zeros((height, width), dtype=bool)
    B_mask = np.zeros((height, width), dtype=bool)
    
    if pattern == 'RGGB':
        R_mask[0::2, 0::2] = True
        G_mask[0::2, 1::2] = True
        G_mask[1::2, 0::2] = True
        B_mask[1::2, 1::2] = True
    elif pattern == 'BGGR':
        B_mask[0::2, 0::2] = True
        G_mask[0::2, 1::2] = True
        G_mask[1::2, 0::2] = True
        R_mask[1::2, 1::2] = True
    elif pattern == 'GRBG':
        G_mask[0::2, 0::2] = True
        R_mask[0::2, 1::2] = True
        B_mask[1::2, 0::2] = True
        G_mask[1::2, 1::2] = True
    elif pattern == 'GBRG':
        G_mask[0::2, 0::2] = True
        B_mask[0::2, 1::2] = True
        R_mask[1::2, 0::2] = True
        G_mask[1::2, 1::2] = True
    
    return R_mask, G_mask, B_mask

def conv_h(x, kernel):
    """Perform horizontal convolution."""
    return convolve1d(x, kernel, axis=1, mode='mirror')

def conv_v(x, kernel):
    """Perform vertical convolution."""
    return convolve1d(x, kernel, axis=0, mode='mirror')

def refining_step(R, G, B, R_mask, G_mask, B_mask, M, debug_save_func=None):
    """
    Perform the refining step on RGB channels based on Menon (2007).
    
    Args:
        R, G, B: Red, Green, Blue channels
        R_mask, G_mask, B_mask: Binary masks for each channel
        M: Directional decision mask (1 for horizontal, 0 for vertical)
        debug_save_func: Function to save debug images
        
    Returns:
        R, G, B: Refined RGB channels
    """
    # Step 1: Update green component
    R_G = R - G
    B_G = B - G
    
    # Apply directional filtering based on M
    B_G_m = np.zeros_like(B_G)
    R_G_m = np.zeros_like(R_G)
    
    B_G_m[B_mask] = np.where(M[B_mask] == 1, 
                             conv_h(B_G, FIR_1D)[B_mask], 
                             conv_v(B_G, FIR_1D)[B_mask])
    
    R_G_m[R_mask] = np.where(M[R_mask] == 1,
                             conv_h(R_G, FIR_1D)[R_mask],
                             conv_v(R_G, FIR_1D)[R_mask])
    
    # Update G at R and B locations
    G = np.where(R_mask, R - R_G_m, G)
    G = np.where(B_mask, B - B_G_m, G)
    
    if debug_save_func is not None:
        debug_save_func(G, "8.0_G_refined_step1")
    
    # Step 2: Update red and blue at green locations
    # Identify row and column patterns
    R_rows = np.zeros(G.shape, dtype=bool)
    R_rows[np.any(R_mask, axis=1), :] = True
    R_cols = np.zeros(G.shape, dtype=bool)
    R_cols[:, np.any(R_mask, axis=0)] = True
    
    B_rows = np.zeros(G.shape, dtype=bool)
    B_rows[np.any(B_mask, axis=1), :] = True
    B_cols = np.zeros(G.shape, dtype=bool)
    B_cols[:, np.any(B_mask, axis=0)] = True
    
    # Recompute color differences
    R_G = R - G
    B_G = B - G
    
    # Update R at G locations
    # In blue rows
    mask = np.logical_and(G_mask, B_rows)
    R_G_m = np.where(mask, conv_v(R_G, K_B_1D), R_G_m)
    R = np.where(mask, G + R_G_m, R)
    
    # In blue columns
    mask = np.logical_and(G_mask, B_cols)
    R_G_m = np.where(mask, conv_h(R_G, K_B_1D), R_G_m)
    R = np.where(mask, G + R_G_m, R)
    
    # Update B at G locations
    # In red rows
    mask = np.logical_and(G_mask, R_rows)
    B_G_m = np.where(mask, conv_v(B_G, K_B_1D), B_G_m)
    B = np.where(mask, G + B_G_m, B)
    
    # In red columns
    mask = np.logical_and(G_mask, R_cols)
    B_G_m = np.where(mask, conv_h(B_G, K_B_1D), B_G_m)
    B = np.where(mask, G + B_G_m, B)
    
    if debug_save_func is not None:
        debug_save_func(R, "8.1_R_refined_step2")
        debug_save_func(B, "8.1_B_refined_step2")
    
    # Step 3: Update red at blue locations and blue at red locations
    R_B = R - B
    
    # Update R at B locations
    R_B_m = np.zeros_like(R_B)
    R_B_m[B_mask] = np.where(M[B_mask] == 1,
                             conv_h(R_B, FIR_1D)[B_mask],
                             conv_v(R_B, FIR_1D)[B_mask])
    R = np.where(B_mask, B + R_B_m, R)
    
    # Update B at R locations
    R_B_m[R_mask] = np.where(M[R_mask] == 1,
                             conv_h(R_B, FIR_1D)[R_mask],
                             conv_v(R_B, FIR_1D)[R_mask])
    B = np.where(R_mask, R - R_B_m, B)
    
    if debug_save_func is not None:
        debug_save_func(R, "8.2_R_final_refined")
        debug_save_func(B, "8.2_B_final_refined")
    
    return R, G, B

def run(img: np.ndarray, pattern: str = 'RGGB', refining_step_enabled: bool = True) -> np.ndarray:
    """
    Processes an image using DDFAPD - Menon (2007) demosaicing algorithm.
    
    Args:
        img (numpy.ndarray): Input 3-channel image with a Bayer mosaic pattern.
        pattern (str): The Bayer pattern ('RGGB', 'BGGR', 'GRBG', 'GBRG').
        refining_step_enabled (bool): Whether to perform the refining step.
        
    Returns:
        numpy.ndarray: Demosaiced 3-channel (RGB) image.
        
    References:
        Menon, D., Andriani, S., & Calvagno, G. (2007).
        Demosaicing With Directional Filtering and a posteriori Decision. 
        IEEE Transactions on Image Processing, 16(1), 132-141.
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
    
    # Convert input to float for processing
    CFA = img.astype(float)
    
    # Create masks for the Bayer pattern
    R_mask, G_mask, B_mask = create_bayer_masks(height, width, pattern)
    
    if debug_mode:
        save_debug_image(CFA, "0.0_input_bayer_img")
        save_debug_image(R_mask, "R_mask", is_mask=True)
        save_debug_image(G_mask, "G_mask", is_mask=True)
        save_debug_image(B_mask, "B_mask", is_mask=True)
    
    # Extract CFA data
    CFA_sum = CFA[:, :, 0] + CFA[:, :, 1] + CFA[:, :, 2]
    
    # Initialize color channels
    R = CFA_sum * R_mask
    G = CFA_sum * G_mask
    B = CFA_sum * B_mask
    
    if debug_mode:
        save_debug_image(R, "1.0_initial_R")
        save_debug_image(G, "1.0_initial_G")
        save_debug_image(B, "1.0_initial_B")
    
    # Step 1: Directional green interpolation
    G_H = np.where(G_mask == 0, 
                   conv_h(CFA_sum, H_0_1D) + conv_h(CFA_sum, H_1_1D), 
                   G)
    G_V = np.where(G_mask == 0, 
                   conv_v(CFA_sum, H_0_1D) + conv_v(CFA_sum, H_1_1D), 
                   G)
    
    if debug_mode:
        save_debug_image(G_H, "2.0_G_horizontal")
        save_debug_image(G_V, "2.0_G_vertical")
    
    # Step 2: Compute color differences
    C_H = np.where(R_mask, R - G_H, 0)
    C_H = np.where(B_mask, B - G_H, C_H)
    
    C_V = np.where(R_mask, R - G_V, 0)
    C_V = np.where(B_mask, B - G_V, C_V)
    
    if debug_mode:
        save_debug_image(C_H, "3.0_C_horizontal")
        save_debug_image(C_V, "3.0_C_vertical")
    
    # Step 3: Compute directional gradients
    D_H = np.abs(C_H - np.pad(C_H, ((0, 0), (0, 2)), mode="reflect")[:, 2:])
    D_V = np.abs(C_V - np.pad(C_V, ((0, 2), (0, 0)), mode="reflect")[2:, :])
    
    if debug_mode:
        save_debug_image(D_H, "4.0_D_horizontal")
        save_debug_image(D_V, "4.0_D_vertical")
    
    # Step 4: Make directional decision
    d_H = convolve(D_H, KERNEL_DECISION_2D, mode='constant')
    d_V = convolve(D_V, np.transpose(KERNEL_DECISION_2D), mode='constant')
    
    if debug_mode:
        save_debug_image(d_H, "5.0_d_horizontal")
        save_debug_image(d_V, "5.0_d_vertical")
    
    # Decision mask: 1 for horizontal, 0 for vertical
    M = (d_V >= d_H).astype(float)
    G = np.where(M == 1, G_H, G_V)
    
    if debug_mode:
        save_debug_image(M, "5.1_decision_mask", is_mask=True)
        save_debug_image(G, "5.2_G_after_decision")
    
    # Step 5: Interpolate R and B
    # Identify row patterns
    R_rows = np.zeros(R.shape, dtype=bool)
    R_rows[np.any(R_mask, axis=1), :] = True
    B_rows = np.zeros(B.shape, dtype=bool)
    B_rows[np.any(B_mask, axis=1), :] = True
    
    # R at G locations
    R = np.where(np.logical_and(G_mask, R_rows),
                 G + conv_h(R, K_B_1D) - conv_h(G, K_B_1D),
                 R)
    R = np.where(np.logical_and(G_mask, B_rows),
                 G + conv_v(R, K_B_1D) - conv_v(G, K_B_1D),
                 R)
    
    # B at G locations
    B = np.where(np.logical_and(G_mask, B_rows),
                 G + conv_h(B, K_B_1D) - conv_h(G, K_B_1D),
                 B)
    B = np.where(np.logical_and(G_mask, R_rows),
                 G + conv_v(B, K_B_1D) - conv_v(G, K_B_1D),
                 B)
    
    if debug_mode:
        save_debug_image(R, "6.0_R_after_G_locations")
        save_debug_image(B, "6.0_B_after_G_locations")
    
    # R at B locations and B at R locations
    R = np.where(np.logical_and(B_rows, B_mask),
                 np.where(M == 1,
                         B + conv_h(R, K_B_1D) - conv_h(B, K_B_1D),
                         B + conv_v(R, K_B_1D) - conv_v(B, K_B_1D)),
                 R)
    
    B = np.where(np.logical_and(R_rows, R_mask),
                 np.where(M == 1,
                         R + conv_h(B, K_B_1D) - conv_h(R, K_B_1D),
                         R + conv_v(B, K_B_1D) - conv_v(R, K_B_1D)),
                 B)
    
    if debug_mode:
        save_debug_image(R, "7.0_R_complete")
        save_debug_image(G, "7.0_G_complete")
        save_debug_image(B, "7.0_B_complete")
    
    # Step 6: Refining step (optional)
    if refining_step_enabled:
        R, G, B = refining_step(R, G, B, R_mask, G_mask, B_mask, M, save_debug_image)
    
    # Step 7: Combine channels
    final_image_float = np.dstack((R, G, B))
    
    if debug_mode:
        save_debug_image(final_image_float, "9.0_final_image_float")
    
    # Round by adding 0.5, then clip to [0, 255] and cast to uint8
    final_image_uint8 = np.clip(final_image_float + 0.5, 0, 255).astype(np.uint8)
    
    if debug_mode:
        save_debug_image(final_image_uint8, "9.1_final_image_uint8")
    
    return final_image_uint8