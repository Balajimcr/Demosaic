import numpy as np
from scipy.signal import convolve2d
from scipy import signal, ndimage
from scipy.ndimage import convolve1d
import os
import imageio
import cv2

debug_mode = True

def save_image(image, filename):
    """Saves an image as a PNG file and corresponding data as a CSV file.

    Args:
        image: The image data to be saved.
        filename: The base filename for both image and CSV files.
    """
    foldername = os.path.join("Data/DLMMSE/")
    os.makedirs(foldername, exist_ok=True)  # Create the folder if it doesn't exist

    # Save the image if debugging is enabled
    if debug_mode:
        imageio.imwrite(foldername + filename + ".png", image.clip(0, 255).astype(np.uint8))

def run(img):
    """
    Main DLMMSE demosaicing function.

    Args:
        img (numpy.ndarray): Input RGB image with a Bayer-like mosaic pattern (R, G, B channels).
    
    Returns:
        numpy.ndarray: Demosaiced 3-channel (RGB) image.
    """

    # Convert the input image to float for numerical processing
    new_img = np.copy(img).astype(float)
    save_image(new_img,"0.0_new_img")

    # Separate the color channels
    R = new_img[:, :, 0]
    G = new_img[:, :, 1]
    B = new_img[:, :, 2]

    # Compute the sum of R, G, and B at each pixel
    # This is used for deriving certain directional interpolations
    S = R + G + B
    save_image(S,"1.0_S")

    # ------------------------------------------------------------
    # Step 1: Interpolate the G channel
    # ------------------------------------------------------------

    # 1.1: Simple directional interpolation (horizontal H and vertical V)
    # Using a 1D convolution across rows and columns to approximate G
    # convolve1d(S, kernel) applies 'kernel' horizontally (along last axis by default)
    H = convolve1d(S, [-1, 2, 2, 2, -1]) / 4
    V = convolve1d(S.T, [-1, 2, 2, 2, -1]).T / 4
    
    save_image(H,"1.1_H")
    save_image(V,"1.1_V")

    # Compute delta_H and delta_V, which are corrections for these interpolations
    # Note the sign inversions at alternating pixel locations to handle the Bayer layout
    delta_H = H - S
    delta_H[0::2, 1::2] = -delta_H[0::2, 1::2]
    delta_H[1::2, 0::2] = -delta_H[1::2, 0::2]

    delta_V = V - S
    delta_V[0::2, 1::2] = -delta_V[0::2, 1::2]
    delta_V[1::2, 0::2] = -delta_V[1::2, 0::2] 

    save_image(delta_H,"1.1_delta_H")
    save_image(delta_V,"1.1_delta_V")

    # 1.2: Apply a Gaussian-like smoothing filter to delta_H and delta_V
    # This is a key step to reduce noise and artifacts in the delta maps
    gaussian_filter = [4, 9, 15, 23, 26, 23, 15, 9, 4]  # 9-tap filter
    gaussian_H = convolve1d(delta_H, gaussian_filter) / np.sum(gaussian_filter)
    gaussian_V = convolve1d(delta_V.T, gaussian_filter).T / np.sum(gaussian_filter)
    
    save_image(gaussian_H,"1.2_gaussian_H")
    save_image(gaussian_V,"1.2_gaussian_V")

    # 1.3: Calculate statistics (mean and variance) for the smoothed delta maps
    # These are used to adaptively refine the interpolation (weighted by variance).
    mean_filter = [1 for _ in range(2*4+1)]  # A 9-tap mean filter
    mean_H = convolve1d(gaussian_H, mean_filter) / np.sum(mean_filter)
    mean_V = convolve1d(gaussian_V.T, mean_filter).T / np.sum(mean_filter)
    
    save_image(mean_H,"1.3_mean_H")
    save_image(mean_V,"1.3_mean_V")

    # var_value_X: the variance of the signal
    var_value_H = convolve1d(np.square(gaussian_H - mean_H), mean_filter) / np.sum(mean_filter) + 1e-10
    var_value_V = convolve1d(np.square(gaussian_V - mean_V).T, mean_filter).T / np.sum(mean_filter) + 1e-10

    save_image(var_value_H,"1.3_var_value_H")
    save_image(var_value_V,"1.3_var_value_V")

    # var_noise_X: the noise variance, estimated from the difference between raw delta and smoothed delta
    var_noise_H = convolve1d(np.square(gaussian_H - delta_H), mean_filter) / np.sum(mean_filter) + 1e-10
    var_noise_V = convolve1d(np.square(gaussian_V - delta_V).T, mean_filter).T / np.sum(mean_filter) + 1e-10
    
    save_image(var_noise_H,"1.3_var_noise_H")
    save_image(var_noise_V,"1.3_var_noise_V")

    # 1.4: Refine the delta maps using the ratio of variance to noise variance
    # new_H and new_V incorporate a learned weighting term for noise suppression
    new_H = mean_H + var_value_H / (var_noise_H + var_value_H) * (delta_H - mean_H)
    new_V = mean_V + var_value_V / (var_noise_V + var_value_V) * (delta_V - mean_V)
    
    save_image(new_H,"1.4_new_H")
    save_image(new_V,"1.4_new_V")

    # 1.5: Combine horizontal and vertical estimates using additional variance-based weights
    var_x_H = np.abs(var_value_H - var_value_H / (var_value_H + var_noise_H)) + 1e-10
    var_x_V = np.abs(var_value_V - var_value_V / (var_value_V + var_noise_V)) + 1e-10

    save_image(var_x_H,"1.5_var_x_H")
    save_image(var_x_V,"1.5_var_x_V")

    w_H = var_x_V / (var_x_H + var_x_V)  # Weight for horizontal direction
    w_V = var_x_H / (var_x_H + var_x_V)  # Weight for vertical direction
    final_result = w_H * new_H + w_V * new_V
    
    save_image(final_result,"1.5_final_result")

    # 1.6: Add the refined delta back to the raw mosaic for the G channel
    # Only update G values in the green pixels' positions:
    # - (0,0) block in each 2x2 cell corresponds to R pixel in a Bayer pattern, so add the delta to R-locations in green plane, etc.
    new_img[0::2, 0::2, 1] = (R + final_result)[0::2, 0::2]
    new_img[1::2, 1::2, 1] = (B + final_result)[1::2, 1::2]
    
    save_image(new_img,"1.6_Add_Delta")

    # Retrieve the updated G channel for subsequent steps
    G = new_img[:, :, 1]
    save_image(G,"1.6_Final_G")

    # ------------------------------------------------------------
    # Step 2: Interpolate R and B channels
    # ------------------------------------------------------------

    # 2.1: Approximate R and B from G using a simple kernel-based interpolation
    # delta_GR -> difference between (G - R) at cross positions
    # delta_GB -> difference between (G - B) at cross positions
    kernel = np.array([[1, 0, 1],
                       [0, 0, 0],
                       [1, 0, 1]])
    delta_GR = convolve2d(G - R, kernel, mode='same') / 4 
    delta_GB = convolve2d(G - B, kernel, mode='same') / 4
    
    save_image(delta_GR,"2.1_delta_GR")
    save_image(delta_GB,"2.1_delta_GB")

    # Update the missing R and B values at the positions initially reserved for them
    new_img[1::2, 1::2, 0] = (G - delta_GR)[1::2, 1::2]  # R in B row/col
    new_img[0::2, 0::2, 2] = (G - delta_GB)[0::2, 0::2]  # B in R row/col

    # 2.2: Further refine R/B interpolation at the complementary positions
    # Using a different kernel shape to capture diagonal neighbors
    R = new_img[:, :, 0]
    B = new_img[:, :, 2]
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])
    delta_GR = convolve2d(G - R, kernel, mode='same') / 4 
    delta_GB = convolve2d(G - B, kernel, mode='same') / 4
    
    save_image(delta_GR,"2.2_delta_GR")
    save_image(delta_GB,"2.2_delta_GB")

    # Fill in the remaining R positions (0,1) & (1,0) blocks for each 2x2 cell
    new_img[0::2, 1::2, 0] = (G - delta_GR)[0::2, 1::2]
    new_img[1::2, 0::2, 0] = (G - delta_GR)[1::2, 0::2]

    # Fill in the remaining B positions (0,1) & (1,0) blocks for each 2x2 cell
    new_img[0::2, 1::2, 2] = (G - delta_GB)[0::2, 1::2]
    new_img[1::2, 0::2, 2] = (G - delta_GB)[1::2, 0::2]

    # Capture final R and B for debugging
    R = new_img[:, :, 0]
    B = new_img[:, :, 2]
    save_image(R,"2.3_R")
    save_image(B,"2.3_B")

    # Return the final demosaiced image
    return (new_img + 0.5).clip(0, 255.5).astype(np.uint8)
