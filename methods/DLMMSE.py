import numpy as np
from scipy.signal import convolve2d
from scipy import signal, ndimage
from scipy.ndimage import convolve1d
import os
import imageio
import cv2

# Ensure the Data/DLMMSE directory exists if debug_mode is True
debug_mode = True
FileName_DLMMSE = "DLMMSE"  # Default filename prefix


def UpdateFileName(FileName):
    """Updates the filename prefix for debug image saves."""
    FileName_DLMMSE = FileName


def save_image(image, stage_name, pattern=None, is_mask=False):
    """Saves an image or mask as a PNG file.

    Args:
        image: The image data to be saved. Can be float or boolean.
        stage_name: A descriptive name for the stage.
        pattern: The Bayer pattern string (optional, for mask filenames).
        is_mask: Boolean, set to True if saving a boolean mask.
    """
    if not debug_mode:
        return

    foldername = os.path.join("Data/DLMMSE/")
    os.makedirs(foldername, exist_ok=True)  # Ensure directory exists here

    # Ensure the image data is in a savable format (0-255 uint8)
    if is_mask:
        # Convert boolean mask to 0 or 255 uint8
        image_to_save = (image * 255).astype(np.uint8)
        filename = f"{FileName_DLMMSE}_{stage_name}"
        if pattern:
            filename = f"{FileName_DLMMSE}_{pattern}_{stage_name}"
        filename += "_mask.png"
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


def run(img, pattern='RGGB'):
    """
    DLMMSE demosaicing function adapted to process different Bayer patterns.

    Args:
        img (numpy.ndarray): Input 3-channel image with a Bayer mosaic pattern.
                            Only specific pixels in each channel contain valid data.
        pattern (str): The Bayer pattern of the input image ('RGGB', 'BGGR', 'GRBG', 'GBRG').

    Returns:
        numpy.ndarray: Demosaiced 3-channel (RGB) image.
    """
    height, width, _ = img.shape

    UpdateFileName(pattern)
    # Validate the input pattern
    supported_patterns = ['RGGB', 'BGGR', 'GRBG', 'GBRG']
    if pattern.upper() not in supported_patterns:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}. Supported patterns are: {supported_patterns}")

    current_pattern = pattern.upper()  # Use uppercase for consistency

    # Convert the input image to float for numerical processing
    # Note: The input 'img' is expected to be a 3-channel image where
    # non-mosaiced pixels are zero (as created by the make_bayer function).
    new_img = np.copy(img).astype(float)
    save_image(new_img, "0.0_input_bayer_img")

    # --- Define Bayer Pattern Slicing Logic ---
    # This dictionary maps pattern names to the slicing tuples for each color component
    # in a 2x2 block (relative to (0,0)).
    pattern_slices = {
        'RGGB': {
            'R': (slice(0, None, 2), slice(0, None, 2)),  # [0::2, 0::2]
            'G1': (slice(0, None, 2), slice(1, None, 2)),  # [0::2, 1::2] (Top-right G)
            'G2': (slice(1, None, 2), slice(0, None, 2)),  # [1::2, 0::2] (Bottom-left G)
            'B': (slice(1, None, 2), slice(1, None, 2)),  # [1::2, 1::2]
        },
        'BGGR': {
            'B': (slice(0, None, 2), slice(0, None, 2)),  # [0::2, 0::2]
            'G1': (slice(0, None, 2), slice(1, None, 2)),  # [0::2, 1::2] (Top-right G)
            'G2': (slice(1, None, 2), slice(0, None, 2)),  # [1::2, 0::2] (Bottom-left G)
            'R': (slice(1, None, 2), slice(1, None, 2)),  # [1::2, 1::2]
        },
        'GRBG': {
            'G1': (slice(0, None, 2), slice(0, None, 2)),  # [0::2, 0::2] (Top-left G)
            'R': (slice(0, None, 2), slice(1, None, 2)),  # [0::2, 1::2]
            'B': (slice(1, None, 2), slice(0, None, 2)),  # [1::2, 0::2]
            'G2': (slice(1, None, 2), slice(1, None, 2)),  # [1::2, 1::2] (Bottom-right G)
        },
        'GBRG': {
            'G1': (slice(0, None, 2), slice(0, None, 2)),  # [0::2, 0::2] (Top-left G)
            'B': (slice(0, None, 2), slice(1, None, 2)),  # [0::2, 1::2]
            'R': (slice(1, None, 2), slice(0, None, 2)),  # [1::2, 0::2]
            'G2': (slice(1, None, 2), slice(1, None, 2)),  # [1::2, 1::2] (Bottom-right G)
        }
    }

    # Get the specific slices for the current pattern
    slices = pattern_slices[current_pattern]

    # --- Generate Bayer Pattern Masks based on the selected pattern ---
    R_mask = np.zeros((height, width), dtype=bool)
    G_mask_01 = np.zeros((height, width), dtype=bool)  # G at pattern's G1 location
    G_mask_10 = np.zeros((height, width), dtype=bool)  # G at pattern's G2 location
    B_mask = np.zeros((height, width), dtype=bool)

    # Assign masks using the slicing dictionary - this is now pattern-dependent
    R_mask[slices['R']] = True
    G_mask_01[slices['G1']] = True
    G_mask_10[slices['G2']] = True
    B_mask[slices['B']] = True

    # Combined G mask (covers both G locations for the pattern)
    G_mask = G_mask_01 | G_mask_10

    # Save the masks for debugging (include pattern name in filename)
    save_image(R_mask, "R_mask", pattern=current_pattern, is_mask=True)
    save_image(G_mask_01, "G_mask_type1", pattern=current_pattern, is_mask=True)
    save_image(G_mask_10, "G_mask_type2", pattern=current_pattern, is_mask=True)
    save_image(G_mask, "G_mask_combined", pattern=current_pattern, is_mask=True)
    save_image(B_mask, "B_mask", pattern=current_pattern, is_mask=True)

    # Separate the color channels from the input (which contains zeros at non-mosaiced locations)
    # These arrays now contain data only at the specific locations determined by the *input* pattern.
    R = new_img[:, :, 0]  # Contains R values only where R was sampled in the input
    G = new_img[:, :, 1]  # Contains G values only where G was sampled in the input
    B = new_img[:, :, 2]  # Contains B values only where B was sampled in the input

    # Save initial separated channels (these show the mosaic pattern clearly per channel)
    save_image(R, "0.2_initial_R")
    save_image(G, "0.2_initial_G")
    save_image(B, "0.2_initial_B")

    # Compute the sum of R, G, and B at each pixel location.
    # 'S' will have non-zero values only at the original mosaic locations for *any* pattern.
    S = R + G + B
    save_image(S, "1.0_S_sum_of_channels")

    # ------------------------------------------------------------
    # Step 1: Interpolate the G channel
    # Estimate G values at R and B locations.
    # ------------------------------------------------------------

    # 1.1: Simple directional interpolation (horizontal H and vertical V)
    # Using a 1D convolution across rows and columns applied to the sum image S.
    kernel_1d_hv = [-1, 2, 2, 2, -1]  # Define the kernel

    # Perform convolution and then divide the result by 4.0
    # Use axis=1 for horizontal convolution across rows
    H = convolve1d(S, kernel_1d_hv, axis=1) / 4.0

    # Use axis=0 for vertical convolution across columns
    V = convolve1d(S, kernel_1d_hv, axis=0) / 4.0

    # Save intermediate results for debugging
    save_image(H, "1.1_H_initial_interpolation")
    save_image(V, "1.1_V_initial_interpolation")

    # Compute delta_H and delta_V, which represent the difference between
    # the interpolated value and the original sum S.
    delta_H = H - S
    delta_V = V - S

    save_image(delta_H, "1.1_delta_H_before_sign_flip")
    save_image(delta_V, "1.1_delta_V_before_sign_flip")

    # Adjust signs of delta_H and delta_V based on pixel color type at the location.
    # The original logic flipped signs at the two G pixel locations (0,1) and (1,0)
    # relative to the top-left of the 2x2 block. Let's re-apply the flip at these *fixed*
    # indices as the original code did, to see if that was the intended behavior
    # for combining with the H-S/V-S delta calculation. This might imply the algorithm
    # implicitly relies on the grid position rather than the color type at that position
    # for the sign flip.

    # Reverting sign flip to original fixed locations [0::2, 1::2] and [1::2, 0::2]
    # regardless of which color is actually there in the current pattern.
    # This seems counter-intuitive for pattern independence, but matches the hardcoded
    # approach in the initial RGGB-only version. Let's test this hypothesis.
    # delta_H_fixed = np.copy(delta_H)
    # delta_V_fixed = np.copy(delta_V)

    # delta_H_fixed[0::2, 1::2] = -delta_H_fixed[0::2, 1::2]
    # delta_H_fixed[1::2, 0::2] = -delta_H_fixed[1::2, 0::2]

    # delta_V_fixed[0::2, 1::2] = -delta_V_fixed[0::2, 1::2]
    # delta_V_fixed[1::2, 0::2] = -delta_V_fixed[1::2, 0::2]

    # save_image(delta_H_fixed,"1.1_delta_H_after_original_sign_flip_logic")
    # save_image(delta_V_fixed,"1.1_delta_V_after_original_sign_flip_logic")

    # Corrected sign flip logic: Flip signs based on actual G pixel locations
    delta_H[G_mask_01] = -delta_H[G_mask_01]
    delta_H[G_mask_10] = -delta_H[G_mask_10]

    delta_V[G_mask_01] = -delta_V[G_mask_01]
    delta_V[G_mask_10] = -delta_V[G_mask_10]

    save_image(delta_H, "1.1_delta_H_after_sign_flip")
    save_image(delta_V, "1.1_delta_V_after_sign_flip")

    # 1.2: Apply a Gaussian-like smoothing filter to delta_H and delta_V.
    gaussian_filter_1d = [4, 9, 15, 23, 26, 23, 15, 9, 4]  # 9-tap filter
    gaussian_filter_1d = np.array(gaussian_filter_1d) / np.sum(gaussian_filter_1d)

    gaussian_H = convolve1d(delta_H, gaussian_filter_1d, axis=1)  # Apply horizontally
    gaussian_V = convolve1d(delta_V, gaussian_filter_1d, axis=0)  # Apply vertically

    save_image(gaussian_H, "1.2_gaussian_smoothed_delta_H")
    save_image(gaussian_V, "1.2_gaussian_smoothed_delta_V")

    # 1.3: Calculate statistics (mean and variance) for the smoothed delta maps.
    mean_filter_1d = np.ones(9) / 9.0  # 9-tap mean filter

    mean_H = convolve1d(gaussian_H, mean_filter_1d, axis=1)
    mean_V = convolve1d(gaussian_V, mean_filter_1d, axis=0)

    save_image(mean_H, "1.3_mean_H")
    save_image(mean_V, "1.3_mean_V")

    # var_value_X: the variance of the signal part of the delta
    var_value_H = convolve1d(np.square(gaussian_H - mean_H), mean_filter_1d, axis=1) + 1e-10
    var_value_V = convolve1d(np.square(gaussian_V - mean_V), mean_filter_1d, axis=0) + 1e-10

    save_image(var_value_H, "1.3_var_value_H")
    save_image(var_value_V, "1.3_var_value_V")

    # var_noise_X: the noise variance, estimated from the difference between raw delta and smoothed delta
    var_noise_H = convolve1d(np.square(delta_H - gaussian_H), mean_filter_1d, axis=1) + 1e-10
    var_noise_V = convolve1d(np.square(delta_V - gaussian_V), mean_filter_1d, axis=0) + 1e-10

    save_image(var_noise_H, "1.3_var_noise_H")
    save_image(var_noise_V, "1.3_var_noise_V")

    # 1.4: Refine the delta maps using the ratio of variance to noise variance (Wiener-like filtering)
    new_H = mean_H + var_value_H / (var_noise_H + var_value_H) * (delta_H - mean_H)
    new_V = mean_V + var_value_V / (var_noise_V + var_value_V) * (delta_V - mean_V)

    save_image(new_H, "1.4_refined_delta_H")
    save_image(new_V, "1.4_refined_delta_V")

    # 1.5: Combine horizontal and vertical refined delta estimates
    var_x_H = np.abs(var_value_H - var_value_H**2 / (var_value_H + var_noise_H)) + 1e-10
    var_x_V = np.abs(var_value_V - var_value_V**2 / (var_value_V + var_noise_V)) + 1e-10

    save_image(var_x_H, "1.5_var_x_H")
    save_image(var_x_V, "1.5_var_x_V")

    w_H = var_x_V / (var_x_H + var_x_V)
    w_V = var_x_H / (var_x_H + var_x_V)

    save_image(w_H, "1.5_weight_H")
    save_image(w_V, "1.5_weight_V")

    # Final combined delta estimate for G interpolation
    final_delta_G = w_H * new_H + w_V * new_V
    save_image(final_delta_G, "1.5_final_delta_G")

    # 1.6: Add the refined delta back to estimate G at R and B locations.
    # The masks R_mask and B_mask now correctly identify the R and B locations
    # for the given input pattern.
    # Estimate G at R locations using the interpolated delta_G and the actual R value
    # new_img[R_mask, 1] = new_img[R_mask, 0] + final_delta_G[R_mask]  # Original line
    # Let's check the channel indices here based on the pattern.
    # In the input 'img', R is always channel 0, G is 1, B is 2.
    # new_img[:,:,0] is the R plane, [:,:,1] is G, [:,:,2] is B.
    # The code is trying to fill the G channel (index 1) at R and B mask locations.
    # new_img[R_mask, 1] should get G data based on R data at R_mask locations.
    # new_img[B_mask, 1] should get G data based on B data at B_mask locations.
    # The channels new_img[R_mask, 0] and new_img[B_mask, 2] correctly access the R and B planes.
    # This part seems correct and pattern independent, as it uses the dynamically created masks.

    # Estimate G at R locations using the interpolated delta_G and the actual R value at R locations
    new_img[R_mask, 1] = new_img[R_mask, 0] + final_delta_G[R_mask]

    # Estimate G at B locations using the interpolated delta_G and the actual B value at B locations
    new_img[B_mask, 1] = new_img[B_mask, 2] + final_delta_G[B_mask]

    # Save the G channel after interpolation at R and B locations
    save_image(new_img[:, :, 1], "1.6_G_interpolated_at_RB")

    # Retrieve the updated G channel for subsequent steps
    G_interpolated = new_img[:, :, 1]
    save_image(G_interpolated, "1.6_Final_G_Channel")

    # ------------------------------------------------------------
    # Step 2: Interpolate R and B channels
    # Estimate R and B values at G locations and complementary color locations.
    # ------------------------------------------------------------

    # Re-extract R and B (these still contain zeros at non-mosaiced locations)
    R_original = new_img[:, :, 0]
    B_original = new_img[:, :, 2]

    # 2.1: Approximate R at B locations and B at R locations (using G-R/G-B differences)
    # Interpolate G-R/G-B differences at the locations where R and B are missing
    # using a kernel averaging diagonal neighbors.
    kernel_diagonal = np.array([[1, 0, 1],
                                [0, 0, 0],
                                [1, 0, 1]]) / 4.0

    # Calculate initial difference maps. These will have valid non-zero values
    # only at the originally sampled locations for R and B.
    # Note: diff_GR is G - R. At R locations (where R is sampled), this is G_interp_at_R - R_actual.
    # At B locations (where B is sampled), this is G_interp_at_B - 0.
    # At G locations (where G is sampled), this is G_actual - 0.
    # The convolution spreads these values.

    diff_GR = G_interpolated - R_original
    diff_GB = G_interpolated - B_original

    save_image(diff_GR, "2.1_diff_GR_initial")
    save_image(diff_GB, "2.1_diff_GB_initial")

    # Interpolate the differences using the diagonal kernel
    delta_GR_interpolated_diag = convolve2d(diff_GR, kernel_diagonal, mode='same')
    delta_GB_interpolated_diag = convolve2d(diff_GB, kernel_diagonal, mode='same')

    save_image(delta_GR_interpolated_diag, "2.1_delta_GR_interpolated_diag")
    save_image(delta_GB_interpolated_diag, "2.1_delta_GB_interpolated_diag")

    # Update the missing R and B values:
    # Estimate R at B locations using G at those locations and interpolated G-R difference
    # R_at_B = G_at_B - (G-R)_interpolated_at_B
    # Use the pattern-specific B_mask to target the B locations
    new_img[B_mask, 0] = G_interpolated[B_mask] - delta_GR_interpolated_diag[B_mask]

    # Estimate B at R locations using G at those locations and interpolated G-B difference
    # B_at_R = G_at_R - (G-B)_interpolated_at_R
    # Use the pattern-specific R_mask to target the R locations
    new_img[R_mask, 2] = G_interpolated[R_mask] - delta_GB_interpolated_diag[R_mask]

    save_image(new_img[:, :, 0], "2.1_R_after_interpolation_at_B")
    save_image(new_img[:, :, 2], "2.1_B_after_interpolation_at_R")

    # 2.2: Further refine R/B interpolation at the remaining G locations
    # These are the locations identified by the combined G_mask for the current pattern.
    # Use a kernel averaging cross neighbors.
    kernel_cross = np.array([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]]) / 4.0

    # Re-calculate difference maps (now R at B, B at R are filled)
    R_partially_filled = new_img[:, :, 0]
    B_partially_filled = new_img[:, :, 2]

    # Note: diff_GR_partial is G - R_partially_filled.
    # At G locations (where R is missing), this is G_actual - 0.
    # At R locations (where R is sampled), this is G_interp_at_R - R_actual.
    # At B locations (where R is interpolated), this is G_interp_at_B - R_interp_at_B.
    # This implies diff_GR_partial represents different things at different locations.
    # The algorithm seems to rely on the filtering and variance weighting to handle this.

    diff_GR_partial = G_interpolated - R_partially_filled
    diff_GB_partial = G_interpolated - B_partially_filled

    save_image(diff_GR_partial, "2.2_diff_GR_partial")
    save_image(diff_GB_partial, "2.2_diff_GB_partial")

    # Interpolate the differences using the cross kernel
    delta_GR_interpolated_cross = convolve2d(diff_GR_partial, kernel_cross, mode='same')
    delta_GB_interpolated_cross = convolve2d(diff_GB_partial, kernel_cross, mode='same')

    save_image(delta_GR_interpolated_cross, "2.2_delta_GR_interpolated_cross")
    save_image(delta_GB_interpolated_cross, "2.2_delta_GB_interpolated_cross")

    # Fill in the remaining R positions at G locations
    # R_at_G = G_at_G - (G-R)_interpolated_at_G
    # Use the pattern-specific G_mask to target the G locations
    new_img[G_mask, 0] = G_interpolated[G_mask] - delta_GR_interpolated_cross[G_mask]

    # Fill in the remaining B positions at G locations
    # B_at_G = G_at_G - (G-B)_interpolated_at_G
    # Use the pattern-specific G_mask to target the G locations
    new_img[G_mask, 2] = G_interpolated[G_mask] - delta_GB_interpolated_cross[G_mask]

    # Capture final R, G, and B channels after all interpolations
    R_final = new_img[:, :, 0]
    G_final = new_img[:, :, 1]  # This is the same as G_interpolated from Step 1.6
    B_final = new_img[:, :, 2]

    save_image(R_final, "2.3_Final_R_Channel")
    save_image(B_final, "2.3_Final_B_Channel")
    # G_final was saved in 1.6, no need to resave unless state changed

    # Return the final demosaiced image
    # Clip values and convert back to uint8
    final_demosaiced_img = (new_img + 0.5).clip(0, 255.5).astype(np.uint8)

    save_image(final_demosaiced_img, "3.0_Final_Demosaiced_Image")

    return final_demosaiced_img