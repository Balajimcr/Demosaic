import numpy as np
from scipy.signal import convolve2d
from scipy import signal, ndimage
from scipy.ndimage import convolve1d
import os
import imageio
import cv2

# Ensure the Data/DLMMSE directory exists if debug_mode is True
debug_mode = True
FileName_DLMMSE = "DLMMSE" # Default filename prefix

def UpdateFileName(filename):
    """Updates the filename prefix for debug image saves."""
    global FileName_DLMMSE
    # Use the base name of the file without extension
    FileName_DLMMSE = os.path.splitext(os.path.basename(filename))[0]

def save_image(image, stage_name, is_mask=False):
    """Saves an image or mask as a PNG file.

    Args:
        image: The image or mask data to be saved. Can be float or boolean.
        stage_name: A descriptive name for the stage.
        is_mask: Boolean, set to True if saving a boolean mask.
    """
    if not debug_mode:
        return

    foldername = os.path.join("Data/DLMMSE/")
    # os.makedirs(foldername, exist_ok=True) # This is already done in the test script

    # Ensure the image data is in a savable format (0-255 uint8)
    if is_mask:
        # Convert boolean mask to 0 or 255 uint8
        image_to_save = (image * 255).astype(np.uint8)
        filename = f"{FileName_DLMMSE}_{stage_name}_mask.png"
    else:
        # Clip float data to 0-255 and convert to uint8
        image_to_save = image.clip(0, 255).astype(np.uint8)
        filename = f"{FileName_DLMMSE}_{stage_name}.png"

    full_path = os.path.join(foldername, filename)
    try:
        # Handle 3-channel vs 1-channel images
        if image_to_save.ndim == 3:
             # OpenCV uses BGR by default, imageio uses RGB. Save as RGB using imageio.
             imageio.imwrite(full_path, image_to_save)
        elif image_to_save.ndim == 2:
             # Save grayscale image
             imageio.imwrite(full_path, image_to_save)
        else:
             print(f"Warning: Cannot save image '{filename}' with unexpected shape {image_to_save.shape}")

        # print(f"Debug image saved: {full_path}") # Optional: Print save path
    except Exception as e:
        print(f"Error saving debug image {full_path}: {e}")


def run(img):
    """
    DLMMSE demosaicing function modified to explicitly process RGGB pattern.

    Args:
        img (numpy.ndarray): Input 3-channel image with RGGB Bayer mosaic pattern.
                             Only specific pixels in each channel contain valid data.

    Returns:
        numpy.ndarray: Demosaiced 3-channel (RGB) image.
    """
    height, width, _ = img.shape

    # Convert the input image to float for numerical processing
    # Note: The input 'img' is expected to be a 3-channel image where
    # non-mosaiced pixels are zero (as created by the make_bayer function).
    new_img = np.copy(img).astype(float)
    save_image(new_img,"0.0_input_bayer_img")

    # --- Define RGGB Bayer Pattern Masks ---
    # These masks help identify the locations of R, G, and B pixels
    # according to the RGGB pattern.
    R_mask = np.zeros((height, width), dtype=bool)
    G_mask_01 = np.zeros((height, width), dtype=bool) # G at (0,1) in 2x2 block
    G_mask_10 = np.zeros((height, width), dtype=bool) # G at (1,0) in 2x2 block
    B_mask = np.zeros((height, width), dtype=bool)

    R_mask[0::2, 0::2] = True
    G_mask_01[0::2, 1::2] = True # Top-right G
    G_mask_10[1::2, 0::2] = True # Bottom-left G
    B_mask[1::2, 1::2] = True
    # Combined G mask
    G_mask = G_mask_01 | G_mask_10

    # Save the masks for debugging
    save_image(R_mask, "0.1_R_mask", is_mask=True)
    save_image(G_mask_01, "0.1_G_mask_01", is_mask=True)
    save_image(G_mask_10, "0.1_G_mask_10", is_mask=True)
    save_image(G_mask, "0.1_G_mask_combined", is_mask=True)
    save_image(B_mask, "0.1_B_mask", is_mask=True)


    # Separate the color channels from the input (which contains zeros at non-mosaiced locations)
    R = new_img[:, :, 0] # Contains R values only at R_mask locations, 0 elsewhere
    G = new_img[:, :, 1] # Contains G values only at G_mask locations, 0 elsewhere
    B = new_img[:, :, 2] # Contains B values only at B_mask locations, 0 elsewhere

    # Save initial separated channels
    save_image(R, "0.2_initial_R")
    save_image(G, "0.2_initial_G")
    save_image(B, "0.2_initial_B")

    # Compute the sum of R, G, and B at each pixel location.
    # Note: 'S' will have non-zero values only at the original mosaic locations.
    S = R + G + B
    save_image(S,"1.0_S_sum_of_channels")

    # ------------------------------------------------------------
    # Step 1: Interpolate the G channel
    # Estimate G values at R and B locations.
    # ------------------------------------------------------------

    # 1.1: Simple directional interpolation (horizontal H and vertical V)
    # Using a 1D convolution across rows and columns applied to the sum image S.
    # The filter [-1, 2, 2, 2, -1] is applied, and the result is divided by 4.
    kernel_1d_hv = [-1, 2, 2, 2, -1] # Define the kernel (as a list is fine for convolve1d)

    # Perform convolution and then divide the result by 4.0
    # Use axis=1 for horizontal convolution across rows
    H = convolve1d(S, kernel_1d_hv, axis=1) / 4.0

    # Use axis=0 for vertical convolution across columns
    V = convolve1d(S, kernel_1d_hv, axis=0) / 4.0

    # Save intermediate results for debugging
    save_image(H,"1.1_H_initial_interpolation")
    save_image(V,"1.1_V_initial_interpolation")

    # Compute delta_H and delta_V, which represent the difference between
    # the interpolated value and the original sum S.
    delta_H = H - S
    delta_V = V - S

    save_image(delta_H,"1.1_delta_H_before_sign_flip")
    save_image(delta_V,"1.1_delta_V_before_sign_flip")

    # Adjust signs of delta_H and delta_V based on pixel location parity.
    # This attempts to make delta (e.g., G-R or G-B) consistent across the grid.
    # For RGGB:
    # At (0,1) and (1,0) locations (Green pixels), delta_H/V is likely G_interpolated - G_actual.
    # At (0,0) (Red) and (1,1) (Blue) locations, delta_H/V is G_interpolated - (R or B).
    # The sign flips here seem designed to align the sign of the delta across these different pixel types.
    # This is a characteristic of the original algorithm's approach.
    delta_H[0::2, 1::2] = -delta_H[0::2, 1::2] # Flip sign at G_01 locations
    delta_H[1::2, 0::2] = -delta_H[1::2, 0::2] # Flip sign at G_10 locations

    delta_V[0::2, 1::2] = -delta_V[0::2, 1::2] # Flip sign at G_01 locations
    delta_V[1::2, 0::2] = -delta_V[1::2, 0::2] # Flip sign at G_10 locations

    save_image(delta_H,"1.1_delta_H_after_sign_flip")
    save_image(delta_V,"1.1_delta_V_after_sign_flip")

    # 1.2: Apply a Gaussian-like smoothing filter to delta_H and delta_V.
    gaussian_filter_1d = [4, 9, 15, 23, 26, 23, 15, 9, 4] # 9-tap filter
    gaussian_filter_1d = np.array(gaussian_filter_1d) / np.sum(gaussian_filter_1d)

    gaussian_H = convolve1d(delta_H, gaussian_filter_1d, axis=1) # Apply horizontally
    gaussian_V = convolve1d(delta_V, gaussian_filter_1d, axis=0) # Apply vertically

    save_image(gaussian_H,"1.2_gaussian_smoothed_delta_H")
    save_image(gaussian_V,"1.2_gaussian_smoothed_delta_V")


    # 1.3: Calculate statistics (mean and variance) for the smoothed delta maps.
    # These statistics are used for adaptive weighting.
    mean_filter_1d = np.ones(9) / 9.0 # 9-tap mean filter

    mean_H = convolve1d(gaussian_H, mean_filter_1d, axis=1)
    mean_V = convolve1d(gaussian_V, mean_filter_1d, axis=0)

    save_image(mean_H,"1.3_mean_H")
    save_image(mean_V,"1.3_mean_V")

    # var_value_X: the variance of the signal part of the delta
    # var_value_H = E[(gaussian_H - mean_H)^2]
    var_value_H = convolve1d(np.square(gaussian_H - mean_H), mean_filter_1d, axis=1) + 1e-10
    var_value_V = convolve1d(np.square(gaussian_V - mean_V), mean_filter_1d, axis=0) + 1e-10

    save_image(var_value_H,"1.3_var_value_H")
    save_image(var_value_V,"1.3_var_value_V")

    # var_noise_X: the noise variance, estimated from the difference between raw delta and smoothed delta
    # var_noise_H = E[(delta_H - gaussian_H)^2]
    var_noise_H = convolve1d(np.square(delta_H - gaussian_H), mean_filter_1d, axis=1) + 1e-10
    var_noise_V = convolve1d(np.square(delta_V - gaussian_V), mean_filter_1d, axis=0) + 1e-10

    save_image(var_noise_H,"1.3_var_noise_H")
    save_image(var_noise_V,"1.3_var_noise_V")


    # 1.4: Refine the delta maps using the ratio of variance to noise variance (Wiener-like filtering)
    # new_H and new_V are the refined delta estimates.
    new_H = mean_H + var_value_H / (var_noise_H + var_value_H) * (delta_H - mean_H)
    new_V = mean_V + var_value_V / (var_noise_V + var_value_V) * (delta_V - mean_V)

    save_image(new_H,"1.4_refined_delta_H")
    save_image(new_V,"1.4_refined_delta_V")


    # 1.5: Combine horizontal and vertical refined delta estimates
    # Weights are based on the "signal-to-noise ratio" of the variance itself.
    # var_x_H/V represents the confidence in the variance estimate.
    var_x_H = np.abs(var_value_H - var_value_H**2 / (var_value_H + var_noise_H)) + 1e-10
    var_x_V = np.abs(var_value_V - var_value_V**2 / (var_value_V + var_noise_V)) + 1e-10

    save_image(var_x_H,"1.5_var_x_H")
    save_image(var_x_V,"1.5_var_x_V")

    # Weights for combining H and V estimates. Higher confidence variance direction gets lower weight.
    # This seems counter-intuitive based on standard Wiener filtering, but follows the provided logic.
    # A more standard approach might use inverse variance as weights.
    # Let's double-check the original paper/source if possible.
    # Assuming the provided logic is correct for THIS algorithm:
    w_H = var_x_V / (var_x_H + var_x_V)  # Weight for horizontal direction
    w_V = var_x_H / (var_x_H + var_x_V)  # Weight for vertical direction

    save_image(w_H,"1.5_weight_H")
    save_image(w_V,"1.5_weight_V")

    # Final combined delta estimate for G interpolation
    final_delta_G = w_H * new_H + w_V * new_V
    save_image(final_delta_G,"1.5_final_delta_G")


    # 1.6: Add the refined delta back to estimate G at R and B locations.
    # According to RGGB pattern:
    # R is at (0,0) + (2i, 2j)
    # G is at (0,1) + (2i, 2j) and (1,0) + (2i, 2j)
    # B is at (1,1) + (2i, 2j)

    # Estimate G at R locations (0::2, 0::2) in the G channel plane
    # G_at_R = R_actual + delta_G_interpolated_at_R
    new_img[R_mask, 1] = new_img[R_mask, 0] + final_delta_G[R_mask] # Use R_mask

    # Estimate G at B locations (1::2, 1::2) in the G channel plane
    # G_at_B = B_actual + delta_G_interpolated_at_B
    new_img[B_mask, 1] = new_img[B_mask, 2] + final_delta_G[B_mask] # Use B_mask

    # Save the G channel after interpolation at R and B locations
    save_image(new_img[:, :, 1], "1.6_G_interpolated_at_RB")

    # Retrieve the updated G channel for subsequent steps
    G_interpolated = new_img[:, :, 1]
    save_image(G_interpolated,"1.6_Final_G_Channel")

    # ------------------------------------------------------------
    # Step 2: Interpolate R and B channels
    # Estimate R and B values at G locations.
    # ------------------------------------------------------------

    # Re-extract R and B (these still contain zeros at non-mosaiced locations)
    R_original = new_img[:, :, 0]
    B_original = new_img[:, :, 2]

    # 2.1: Approximate R and B at B and R locations respectively (using G-R/G-B differences)
    # Interpolate G-R at B locations (1::2, 1::2) and G-B at R locations (0::2, 0::2)
    # using a kernel averaging diagonal neighbors.
    kernel_diagonal = np.array([[1, 0, 1],
                                [0, 0, 0],
                                [1, 0, 1]]) / 4.0

    # Calculate initial difference maps (will have valid non-zero values where G or R/B are known)
    diff_GR = G_interpolated - R_original
    diff_GB = G_interpolated - B_original

    save_image(diff_GR, "2.1_diff_GR_initial")
    save_image(diff_GB, "2.1_diff_GB_initial")

    # Interpolate the differences using the diagonal kernel
    delta_GR_interpolated_diag = convolve2d(diff_GR, kernel_diagonal, mode='same')
    delta_GB_interpolated_diag = convolve2d(diff_GB, kernel_diagonal, mode='same')

    save_image(delta_GR_interpolated_diag,"2.1_delta_GR_interpolated_diag")
    save_image(delta_GB_interpolated_diag,"2.1_delta_GB_interpolated_diag")


    # Update the missing R and B values at the complementary positions:
    # Estimate R at B locations (1::2, 1::2) using G at those locations and interpolated G-R difference
    # R_at_B = G_at_B - (G-R)_interpolated_at_B
    new_img[B_mask, 0] = G_interpolated[B_mask] - delta_GR_interpolated_diag[B_mask] # Use B_mask

    # Estimate B at R locations (0::2, 0::2) using G at those locations and interpolated G-B difference
    # B_at_R = G_at_R - (G-B)_interpolated_at_R
    new_img[R_mask, 2] = G_interpolated[R_mask] - delta_GB_interpolated_diag[R_mask] # Use R_mask

    save_image(new_img[:, :, 0], "2.1_R_after_interpolation_at_B")
    save_image(new_img[:, :, 2], "2.1_B_after_interpolation_at_R")


    # 2.2: Further refine R/B interpolation at the remaining G locations (0::2, 1::2) and (1::2, 0::2)
    # using a different kernel shape to capture cross neighbors.
    kernel_cross = np.array([[0, 1, 0],
                             [1, 0, 1],
                             [0, 1, 0]]) / 4.0

    # Re-calculate difference maps (now G, R at B, B at R are filled)
    R_partially_filled = new_img[:, :, 0]
    B_partially_filled = new_img[:, :, 2]

    diff_GR_partial = G_interpolated - R_partially_filled
    diff_GB_partial = G_interpolated - B_partially_filled

    save_image(diff_GR_partial, "2.2_diff_GR_partial")
    save_image(diff_GB_partial, "2.2_diff_GB_partial")


    # Interpolate the differences using the cross kernel
    delta_GR_interpolated_cross = convolve2d(diff_GR_partial, kernel_cross, mode='same')
    delta_GB_interpolated_cross = convolve2d(diff_GB_partial, kernel_cross, mode='same')

    save_image(delta_GR_interpolated_cross,"2.2_delta_GR_interpolated_cross")
    save_image(delta_GB_interpolated_cross,"2.2_delta_GB_interpolated_cross")


    # Fill in the remaining R positions at G locations (0::2, 1::2) and (1::2, 0::2)
    # R_at_G = G_at_G - (G-R)_interpolated_at_G
    new_img[G_mask, 0] = G_interpolated[G_mask] - delta_GR_interpolated_cross[G_mask] # Use G_mask

    # Fill in the remaining B positions at G locations (0::2, 1::2) and (1::2, 0::2)
    # B_at_G = G_at_G - (G-B)_interpolated_at_G
    new_img[G_mask, 2] = G_interpolated[G_mask] - delta_GB_interpolated_cross[G_mask] # Use G_mask


    # Capture final R, G, and B channels after all interpolations
    R_final = new_img[:, :, 0]
    G_final = new_img[:, :, 1] # This is the same as G_interpolated from Step 1.6
    B_final = new_img[:, :, 2]

    save_image(R_final,"2.3_Final_R_Channel")
    save_image(B_final,"2.3_Final_B_Channel")
    # G_final was saved in 1.6, no need to resave unless state changed

    # Return the final demosaiced image
    # Clip values and convert back to uint8
    final_demosaiced_img = (new_img + 0.5).clip(0, 255.5).astype(np.uint8)

    save_image(final_demosaiced_img, "3.0_Final_Demosaiced_Image")

    return final_demosaiced_img