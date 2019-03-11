import numpy as np
import cv2

def color_gradient_thresh(img, s_thresh, l_thresh, sobel_kernel, mag_thresh,graddir_thresh):
              
    # Take both Sobel x and y gradients
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8)            
    # threshold gradient magnitude
    gradmag_binary = np.zeros_like(gradmag)
    gradmag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])]
    
    # Take gradient orientation threshold 
    graddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    graddir_binary = np.zeros_like(graddir)
    graddir_binary[(graddir >= graddir_thresh[0]) & (graddir <= graddir_thresh[1])] = 1
    
    # Take both saturation and lightness channel for thresholding 
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    #threshold lightness
    l_binary = np.zeros_like(l_channel) 
    l_binary[(l_channel > l_thresh[0]) & (l_channel < l_thresh[1])] = 1  
    #threshold saturation 
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    binary = np.zeros_like(s_binary)
    binary[((l_binary == 1) | (s_binary == 1) | (gradmag_binary==1) | (graddir_thresh == 1))] = 1
        
    return binary