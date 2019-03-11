import numpy as np
from collections import deque  
import cv2
import matplotlib.pyplot as plt

import cameraclibration 
import color_gradient_thresh 
import perspective_transform
import sliding_window 
import search_from_prior
import curvature 
import draw_lane

class Line():
    def __init__(self,n):
        # camera calibration
        self.calibration = False 
        self.camera_cali_matrix=None
        self.camera_dist=None
        
        # was the line detected in the last iteration?
        self.detected = False
        self.fit_tolerance = np.array([0.5,0.5,0.5])
        self.curvature_tolerance = 1.8
        # number of data to be stored 
        self.n=n
        
        #average x values of the fitted line over the last n iterations
        self.n_leftx = deque([],maxlen = n)
        self.n_rightx = deque([],maxlen = n)
        self.avg_leftx = None
        self.avg_rightx = None
        
        #polynomial coefficients averaged over the last n iterations
        self.n_left_fit = deque([],maxlen = n)
        self.n_right_fit = deque([],maxlen = n)
        self.avg_left_fit = None
        self.avg_right_fit = None
        
        #radius of curvature of the line in the last n iterations
        self.radius_of_curvature = deque([],maxlen = n)
        self.avg_curvature = None
        
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        
        #difference in fit coefficients between last n average and new fits
        self.left_diffs = np.array([0,0,0], dtype='float')
        self.right_diffs = np.array([0,0,0], dtype='float')
            
    def camera_calibration_undistortion(self,img):
        #camera calibration
        if self.calibration==False:
            self.camera_cali_matrix,self.camera_dist = cameraclibration.cameraclibration()
            self.calibration = True    
        #undistort image 
        undist = cv2.undistort(img, self.camera_cali_matrix, self.camera_dist, None, self.camera_cali_matrix)
        
        return undist
    
    def average_leftx(self):
        X = self.n_leftx
        summation = 0
        if len(X)>0:
            for x in X:
                summation +=np.array(x)
            self.avg_leftx = summation/len(X)
    
    def average_rightx(self):
        X = self.n_rightx
        summation = 0
        if len(X)>0:
            for x in X:
                summation +=np.array(x)
            self.avg_rightx = summation/len(X)    
        
    def average_left_fit(self):
        coeffs = self.n_left_fit
        summation = 0
        if len(coeffs)>0:
            for coeff in coeffs:
                summation +=np.array(coeff)
            self.avg_left_fit=summation/len(coeffs)
    
    def average_right_fit(self):
        coeffs = self.n_right_fit
        summation = 0
        if len(coeffs)>0:
            for coeff in coeffs:
                summation +=np.array(coeff)
            self.avg_right_fit=summation/len(coeffs)
        
    def average_curvature(self):
        curvatures = self.radius_of_curvature
        summation  = 0
        if len(curvatures) > 0:
            for radius in curvatures:
                summation += np.array(radius)
            self.avg_curvature = summation/len(curvatures)
    
    def sanity_check(self,left_fit,right_fit,radius):
         self.left_diffs = abs(np.subtract(self.avg_left_fit,left_fit))
         self.right_diffs = abs(np.subtract(self.avg_right_fit,right_fit))
         left_relative_delta = self.left_diffs/self.avg_left_fit
         right_relative_delta = self.right_diffs/self.avg_right_fit
         if (left_relative_delta > self.fit_tolerance).all() | (right_relative_delta > self.fit_tolerance).all():
             print ('Fit coefficients are too far off')
             self.detected = False 
         if abs(radius-self.avg_curvature)/self.avg_curvature > self.curvature_tolerance:
             print ('Line curvature is too far off')
             self.detected = False 
        
    def update(self, left_fitx, right_fitx, left_fit, right_fit, radius):
        #store the last 5 iterations' left and right lines positions 
        self.n_leftx.appendleft(left_fitx)
        self.n_rightx.appendleft(right_fitx)
        
        #store the last 5 iterations' polynomial parameters 
        self.n_left_fit.appendleft(left_fit)
        self.n_right_fit.appendleft(right_fit)
        if len(self.n_left_fit) == 1:
            self.avg_left_fit = left_fit
        if len(self.n_right_fit) == 1:
            self.avg_right_fit = right_fit
        
        #store the last 5 iterations' curvature and calculate the average
        line.radius_of_curvature.appendleft(radius)
        if len(self.radius_of_curvature) == 1:
            self.avg_curvature = radius
            
        self.sanity_check(left_fit,right_fit,radius)
        # if the new polynomial parameters or curvature are too far off, remove that new data 
        if self.detected == False:  
            self.n_leftx.popleft()
            self.n_rightx.popleft()
            self.average_leftx()
            self.average_rightx()
            
            self.n_left_fit.popleft()
            self.n_right_fit.popleft()
            self.average_left_fit()
            self.average_right_fit()
            
            self.radius_of_curvature.popleft()
            self.average_curvature()
        else:
            self.average_leftx()
            self.average_rightx()
            self.average_left_fit()
            self.average_right_fit()
            self.average_curvature()
                            
def pipeline(img):
    
    # camera calibration and image undistortion
    undist = line.camera_calibration_undistortion(img)
        
    #color and gradient threshold 
    binary_output = color_gradient_thresh.color_gradient_thresh(undist, s_thresh=(130,255), 
                                    l_thresh=(225,255), sobel_kernel=3, mag_thresh=(100, 255),graddir_thresh=(0, np.pi/2))
    
    #perspective transform 
    srcpoints = np.float32([[585,460],[203,720],[1127,720],[695,460]])
    dstpoints = np.float32([[320,0],[320,720],[960,720],[960,0]])
    binary_warped=perspective_transform.corners_unwarp(binary_output, srcpoints, dstpoints)
    
    # Choose either sliding window or search from prior based on sanity check           
    if line.detected==False: 
        #sliding window and find the initial polynomial parameters 
        left_fit,right_fit,left_fitx,right_fitx=sliding_window.fit_polynomial(binary_warped)
        line.detected=True   
        #search from prior
        left_fit, right_fit, left_fitx, right_fitx = search_from_prior.search_around_poly(binary_warped,left_fit,right_fit)
    else:    
        #search from prior
        left_fit, right_fit, left_fitx, right_fitx = search_from_prior.search_around_poly(binary_warped,line.avg_left_fit,line.avg_right_fit) 
     
    #measure curvature 
    y_eval=undist.shape[0]
    left_radius, right_radius = curvature.measure_curvature_real(left_fit,right_fit,y_eval)
    radius=(left_radius+right_radius)/2.0     
         
    line.update(left_fitx, right_fitx, left_fit, right_fit, radius)
    
    # draw lane mark 
    lane_marked_img = draw_lane.draw_lane_mark(binary_warped,line.avg_leftx,line.avg_rightx)
         
    #perspective transform back to original perspective 
    binary_original = perspective_transform.corners_unwarp(lane_marked_img,dstpoints,srcpoints)
    
    #combine original image with lane marked image 
    result = cv2.addWeighted(undist, 1, binary_original, 0.3, 0)
    
    #measure the car center relative to lane center 
    xm_per_pix = 3.7/378 # meters per pixel in x dimension
    line.line_base_pos=((left_fitx[-1] + right_fitx[-1])/2.0-undist.shape[1]/2.0)*xm_per_pix
    if line.line_base_pos > 0:
        direction = 'left'
    elif line.line_base_pos < 0:
        direction = 'right'
    # print radius and distance information 
    font = cv2.FONT_HERSHEY_SIMPLEX
    str1 = str('Radius of curvature ' + '{:04.2f}'.format(line.avg_curvature/1000) + 'km')
    cv2.putText(result,str1,(400,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    str2 = str('{:04.2f}'.format(abs(line.line_base_pos)) + 'm ' + direction + ' of center')
    cv2.putText(result,str2,(400,80), font, 1,(255,255,255),2,cv2.LINE_AA)
    
    return result

    
#line=Line(5)
#img = cv2.imread('../test_images/test5.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#plt.figure(1, figsize=(12, 9))
#plt.imshow(img)
#plt.title('Original Image')
#result=pipeline(img)
#plt.figure(2,figsize=(12, 9))
#
#plt.imshow(result)


#Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

line=Line(5)
out_dir='../'
output = out_dir+'processed_project_video.mp4'
clip = VideoFileClip("../project_video.mp4")
out_clip = clip.fl_image(pipeline) 
out_clip.write_videofile(output, audio=False)
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output))



