import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d

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


def run(img: np.ndarray) -> np.ndarray:
    """
    Processes an image using a multi-step algorithm involving convolutions and
    channel interpolations, likely for color artifact reduction or demosaicing refinement.

    The indexing for channel updates (e.g., `new_G[0::2, 0::2]`) assumes a Bayer-like
    grid logic (e.g., R at (even,even), B at (odd,odd) for a typical RGGB pattern).
    The input 'img' is assumed to be a 3-channel RGB image.
    """
    S = img.astype(float)
    R, G, B = S[:, :, 0], S[:, :, 1], S[:, :, 2]

    # Initialize new color channels by copying the original ones.
    # These will be selectively updated throughout the process.
    new_R, new_G, new_B = np.copy(R), np.copy(G), np.copy(B)

    sum_RGB = R + G + B

    # Step 1: Interpolate color information horizontally and vertically
    # scipy.ndimage.convolve1d's default mode is 'reflect'.
    H_interpolated = convolve1d(sum_RGB, INTERP_KERNEL_1D, axis=1, mode='reflect')

    # Create temporary channel versions based on horizontal interpolation.
    # These assignments update specific subgrids based on a Bayer-like pattern.
    G_H, R_H, B_H = np.copy(G), np.copy(R), np.copy(B)
    G_H[0::2, 0::2] = H_interpolated[0::2, 0::2]  # e.g., G sites at (even,even)
    G_H[1::2, 1::2] = H_interpolated[1::2, 1::2]  # e.g., G sites at (odd,odd)
    R_H[0::2, 1::2] = H_interpolated[0::2, 1::2]  # e.g., B sites at (even,odd)
    B_H[1::2, 0::2] = H_interpolated[1::2, 0::2]  # e.g., R sites at (odd,even)

    V_interpolated = convolve1d(sum_RGB, INTERP_KERNEL_1D, axis=0, mode='reflect')

    # Create temporary channel versions based on vertical interpolation.
    G_V, R_V, B_V = np.copy(G), np.copy(R), np.copy(B)
    G_V[0::2, 0::2] = V_interpolated[0::2, 0::2]
    G_V[1::2, 1::2] = V_interpolated[1::2, 1::2]
    # Note: R_V and B_V update locations are "crossed" compared to R_H, B_H update locations.
    R_V[1::2, 0::2] = V_interpolated[1::2, 0::2]  # Corresponds to B_H update sites
    B_V[0::2, 1::2] = V_interpolated[0::2, 1::2]  # Corresponds to R_H update sites

    # Step 2: Compute delta values (differences) and their gradients
    delta_H = G_H - R_H - B_H
    delta_V = G_V - R_V - B_V

    # Gradient calculation (absolute difference) using convolve1d. Default mode 'reflect'.
    D_H = np.absolute(convolve1d(delta_H, GRAD_KERNEL_1D, axis=1, mode='reflect'))
    D_V = np.absolute(convolve1d(delta_V, GRAD_KERNEL_1D, axis=0, mode='reflect'))

    # Step 3: Compute directional weight coefficients (eq.6 in paper)
    # scipy.signal.convolve2d's default boundary='fill', fillvalue=0.0.
    W_W = convolve2d(D_H, KERNEL_W_W_2D, mode='same', boundary='fill', fillvalue=0.0)
    W_E = convolve2d(D_H, KERNEL_W_E_2D, mode='same', boundary='fill', fillvalue=0.0)
    W_N = convolve2d(D_V, KERNEL_W_N_2D, mode='same', boundary='fill', fillvalue=0.0)
    W_S = convolve2d(D_V, KERNEL_W_S_2D, mode='same', boundary='fill', fillvalue=0.0)

    # Process weights: apply epsilon for zero weights and compute inverse square.
    # This matches the original code's sequential modification of W_X variables.
    W_W[W_W == 0] = WEIGHT_EPSILON; W_W = 1.0 / np.square(W_W)
    W_E[W_E == 0] = WEIGHT_EPSILON; W_E = 1.0 / np.square(W_E)
    W_N[W_N == 0] = WEIGHT_EPSILON; W_N = 1.0 / np.square(W_N)
    W_S[W_S == 0] = WEIGHT_EPSILON; W_S = 1.0 / np.square(W_S)

    W_T = W_W + W_E + W_N + W_S  # Total weight

    # Step 4: Compute final delta values (eq.5 in paper)
    each_delta_component = []
    for c_pattern_offset in range(2):  # Corresponds to 'c' in the original code (0 or 1)
        # Create sparsely populated delta maps based on the current offset
        current_delta_H = np.zeros_like(delta_H)
        current_delta_V = np.zeros_like(delta_V)

        current_delta_H[c_pattern_offset::2, :] = delta_H[c_pattern_offset::2, :]
        current_delta_V[:, c_pattern_offset::2] = delta_V[:, c_pattern_offset::2]

        # Convolutions (convolve1d default mode 'reflect')
        # V1, V2 are from vertical processing (axis=0) of current_delta_V
        V1_N_component = convolve1d(current_delta_V, KERNEL_F_FORWARD_1D, axis=0, mode='reflect')
        V2_S_component = convolve1d(current_delta_V, KERNEL_F_BACKWARD_1D, axis=0, mode='reflect')
        # V3, V4 are from horizontal processing (axis=1) of current_delta_H
        V3_E_component = convolve1d(current_delta_H, KERNEL_F_FORWARD_1D, axis=1, mode='reflect')
        V4_W_component = convolve1d(current_delta_H, KERNEL_F_BACKWARD_1D, axis=1, mode='reflect')

        # Combine weighted components. W_T should not be zero due to WEIGHT_EPSILON.
        final_delta = (V1_N_component * W_N + V2_S_component * W_S +
                       V3_E_component * W_E + V4_W_component * W_W) / W_T
        each_delta_component.append(final_delta)

    delta_GR, delta_GB = each_delta_component[0], each_delta_component[1]

    # Step 5: Recover G channel (Eq.8 in paper)
    # Assuming a Bayer pattern (e.g., RGGB):
    # (0::2, 0::2) are R-pixel native locations. delta_GR is G-R.
    # (1::2, 1::2) are B-pixel native locations. delta_GB is G-B.
    # Update G at R-locations: new_G = R_original + (G-R)_estimated
    new_G[0::2, 0::2] = R[0::2, 0::2] + delta_GR[0::2, 0::2]
    # Update G at B-locations: new_G = B_original + (G-B)_estimated
    new_G[1::2, 1::2] = B[1::2, 1::2] + delta_GB[1::2, 1::2]
    # Original G samples (at G-native locations) in new_G remain unchanged from the initial copy.

    # Step 6: Recover R at B-locations and B at R-locations (Eq.9 in paper)
    # Update R at B-locations (e.g., (1::2, 1::2) for B in RGGB pattern)
    # new_R[B_loc] = new_G[B_loc] - convolved_version_of(delta_GR)
    convolved_delta_for_R_at_B = convolve2d(delta_GR, KERNEL_PRB_2D, mode='same', boundary='fill', fillvalue=0.0) / SUM_KERNEL_PRB_2D
    new_R[1::2, 1::2] = (new_G - convolved_delta_for_R_at_B)[1::2, 1::2]

    # Update B at R-locations (e.g., (0::2, 0::2) for R in RGGB pattern)
    # new_B[R_loc] = new_G[R_loc] - convolved_version_of(delta_GB)
    convolved_delta_for_B_at_R = convolve2d(delta_GB, KERNEL_PRB_2D, mode='same', boundary='fill', fillvalue=0.0) / SUM_KERNEL_PRB_2D
    new_B[0::2, 0::2] = (new_G - convolved_delta_for_B_at_R)[0::2, 0::2]
    # Native R samples in new_R and native B samples in new_B remain unchanged.

    # Step 7: Recover R and B at G-locations (Eq.10 in paper)
    # G-locations are e.g., (0::2, 1::2) and (1::2, 0::2) for G in RGGB pattern.

    # Recover R at G-locations:
    # R_at_G = G_original - K_conv * (G_interpolated - R_partially_interpolated)
    diff_G_newR = new_G - new_R
    R_at_G_correction = convolve2d(diff_G_newR, KERNEL_RB_G_2D, mode='same', boundary='fill', fillvalue=0.0) / SUM_KERNEL_RB_G_2D
    R_interp_at_G_locations = G - R_at_G_correction  # Uses original G channel values

    new_R[0::2, 1::2] = R_interp_at_G_locations[0::2, 1::2]
    new_R[1::2, 0::2] = R_interp_at_G_locations[1::2, 0::2]

    # Recover B at G-locations:
    diff_G_newB = new_G - new_B
    B_at_G_correction = convolve2d(diff_G_newB, KERNEL_RB_G_2D, mode='same', boundary='fill', fillvalue=0.0) / SUM_KERNEL_RB_G_2D
    B_interp_at_G_locations = G - B_at_G_correction  # Uses original G channel values

    new_B[0::2, 1::2] = B_interp_at_G_locations[0::2, 1::2]
    new_B[1::2, 0::2] = B_interp_at_G_locations[1::2, 0::2]

    # Step 8: Finalize image construction
    final_image_float = np.dstack((new_R, new_G, new_B))

    # Round by adding 0.5 (for float to int conversion behavior), then clip to [0, 255] and cast to uint8.
    final_image_uint8 = np.clip(final_image_float + 0.5, 0, 255).astype(np.uint8)

    return final_image_uint8