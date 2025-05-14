import numpy as np
from scipy.ndimage import convolve
import os
import imageio

# Global configurations
debug_mode = False
FileName_Malvar = "Malvar"  # Default filename prefix

# --- Algorithmic Constants ---
# Kernels are precomputed for efficiency and consistency with GBTF.py style

# For Green at Red/Blue locations
KERNEL_GR_GB_2D = np.array([
    [0.0, 0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 0.0, 0.0],
    [-1.0, 2.0, 4.0, 2.0, -1.0],
    [0.0, 0.0, 2.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0, 0.0],
], dtype=float) / 8.0

# For Red/Blue at Green locations (horizontal/vertical cases)
KERNEL_Rg_RB_Bg_BR_2D = np.array([
    [0.0, 0.0, 0.5, 0.0, 0.0],
    [0.0, -1.0, 0.0, -1.0, 0.0],
    [-1.0, 4.0, 5.0, 4.0, -1.0],
    [0.0, -1.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.5, 0.0, 0.0],
], dtype=float) / 8.0

# Transpose for the other direction
KERNEL_Rg_BR_Bg_RB_2D = KERNEL_Rg_RB_Bg_BR_2D.T

# For Red at Blue and Blue at Red locations
KERNEL_Rb_BB_Br_RR_2D = np.array([
    [0.0, 0.0, -1.5, 0.0, 0.0],
    [0.0, 2.0, 0.0, 2.0, 0.0],
    [-1.5, 0.0, 6.0, 0.0, -1.5],
    [0.0, 2.0, 0.0, 2.0, 0.0],
    [0.0, 0.0, -1.5, 0.0, 0.0],
], dtype=float) / 8.0

def update_filename(pattern):
    """Updates the filename prefix for debug image saves."""
    global FileName_Malvar
    FileName_Malvar = pattern

def save_image(image, stage_name, pattern=None, is_mask=False):
    """Saves an image or mask as a PNG file if debug_mode is enabled."""
    if not debug_mode:
        return

    # Create debug directory if it doesn't exist
    foldername = os.path.join("Data/Malvar/")
    os.makedirs(foldername, exist_ok=True)

    # Prepare the image for saving
    if is_mask:
        image_to_save = (image * 255).astype(np.uint8)
        filename = f"{FileName_Malvar}_{stage_name}"
        if pattern:
            filename = f"{FileName_Malvar}_{pattern}_{stage_name}"
        filename += "_mask.png"
    else:
        image_to_save = image.clip(0, 255).astype(np.uint8)
        filename = f"{FileName_Malvar}_{stage_name}.png"

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

def run(img: np.ndarray, pattern: str = 'RGGB') -> np.ndarray:
    """
    Processes an image using Malvar (2004) demosaicing algorithm.
    
    Args:
        img (numpy.ndarray): Input 3-channel image with a Bayer mosaic pattern.
        pattern (str): The Bayer pattern ('RGGB', 'BGGR', 'GRBG', 'GBRG').
        
    Returns:
        numpy.ndarray: Demosaiced 3-channel (RGB) image.
        
    References:
        Malvar, H. S., He, L.-W., Cutler, R., & Way, O. M. (2004). 
        High-Quality Linear Interpolation for Demosaicing of Bayer-Patterned Color Images. 
        International Conference of Acoustic, Speech and Signal Processing, 5-8.
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
    
    # Extract individual channel data from the CFA
    # In the CFA, each pixel location has only one color value
    # We need to extract the sum of all channels to get the actual CFA values
    CFA_sum = CFA[:, :, 0] + CFA[:, :, 1] + CFA[:, :, 2]
    
    # Initialize color channels with existing values
    R = CFA_sum * R_mask
    G = CFA_sum * G_mask
    B = CFA_sum * B_mask
    
    if debug_mode:
        save_debug_image(R, "1.0_initial_R")
        save_debug_image(G, "1.0_initial_G")
        save_debug_image(B, "1.0_initial_B")
    
    # Step 1: Interpolate Green channel at Red and Blue locations
    G_interpolated = convolve(CFA_sum, KERNEL_GR_GB_2D, mode='mirror')
    G = np.where(np.logical_or(R_mask, B_mask), G_interpolated, G)
    
    if debug_mode:
        save_debug_image(G, "2.0_G_after_interpolation")
    
    # Step 2: Calculate convolutions for Red and Blue interpolation
    RBg_RBBR = convolve(CFA_sum, KERNEL_Rg_RB_Bg_BR_2D, mode='mirror')
    RBg_BRRB = convolve(CFA_sum, KERNEL_Rg_BR_Bg_RB_2D, mode='mirror')
    RBgr_BBRR = convolve(CFA_sum, KERNEL_Rb_BB_Br_RR_2D, mode='mirror')
    
    # Create row and column indicators for Red and Blue locations
    # Red rows: rows that contain red pixels
    R_rows = np.zeros_like(R, dtype=bool)
    R_rows[np.any(R_mask, axis=1), :] = True
    # Red columns: columns that contain red pixels
    R_cols = np.zeros_like(R, dtype=bool)
    R_cols[:, np.any(R_mask, axis=0)] = True
    
    # Blue rows: rows that contain blue pixels
    B_rows = np.zeros_like(B, dtype=bool)
    B_rows[np.any(B_mask, axis=1), :] = True
    # Blue columns: columns that contain blue pixels
    B_cols = np.zeros_like(B, dtype=bool)
    B_cols[:, np.any(B_mask, axis=0)] = True
    
    if debug_mode:
        save_debug_image(R_rows, "R_rows", is_mask=True)
        save_debug_image(R_cols, "R_cols", is_mask=True)
        save_debug_image(B_rows, "B_rows", is_mask=True)
        save_debug_image(B_cols, "B_cols", is_mask=True)
    
    # Step 3: Interpolate Red and Blue channels
    # Red at Green locations (in Blue rows/Red columns and Red rows/Blue columns)
    R = np.where(np.logical_and(R_rows, B_cols), RBg_RBBR, R)
    R = np.where(np.logical_and(B_rows, R_cols), RBg_BRRB, R)
    
    # Blue at Green locations (in Blue rows/Red columns and Red rows/Blue columns)
    B = np.where(np.logical_and(B_rows, R_cols), RBg_RBBR, B)
    B = np.where(np.logical_and(R_rows, B_cols), RBg_BRRB, B)
    
    # Red at Blue locations and Blue at Red locations
    R = np.where(np.logical_and(B_rows, B_cols), RBgr_BBRR, R)
    B = np.where(np.logical_and(R_rows, R_cols), RBgr_BBRR, B)
    
    if debug_mode:
        save_debug_image(R, "3.0_final_R")
        save_debug_image(G, "3.0_final_G")
        save_debug_image(B, "3.0_final_B")
    
    # Step 4: Combine channels into final image
    final_image_float = np.dstack((R, G, B))
    
    if debug_mode:
        save_debug_image(final_image_float, "4.0_final_image_float")
    
    # Round by adding 0.5, then clip to [0, 255] and cast to uint8
    final_image_uint8 = np.clip(final_image_float + 0.5, 0, 255).astype(np.uint8)
    
    if debug_mode:
        save_debug_image(final_image_uint8, "4.1_final_image_uint8")
    
    return final_image_uint8