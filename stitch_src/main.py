import cv2
import numpy as np
import sys
from time import time

"""
Keypoint: data structure for salient point detectors. 2d position, scale, orientation and 
some other parameters. The keypoint neighborhood is then analyzed by another algorithm that
builds a descriptor (usually represented as a feature vector).

Dmath:
    queryIdx : index in the descriptor in the query image. Corresponds to the index of the keypoint
                descriptor in the list of keypoints/descriptor for the query image. First image
    trainIdx : the same as the queryIdx but for the second image.
    distance :  distance of dissimilarity measure between the descriptors of the qiery and train
                keypoints that resulted in this match. the smaller the distance, the better the
                match. You can use this valie to filter of rank matches based on theur equality.
    imgIdx   : index of the image to which the train descriptor belongs. ID

findHomography : finds homography matrix that describes the transformation between two sets of 
corresponding points in two images. This function is often used in computer vision applications,
such as image stitching, to fing the geometic transformation between two images taken from different
viewpoints or angles.
cv2.RANSAC      : glag that specifies the robust estimation method to be used for finding the 
homography matrix. RANSAC(random sample concensus) is a common method used for robust estimation
of outliers.
H               : this is the homography matrix. represents the transformation that maps points
from the current image to the previous image. When you apply this tranformation to the points in 
the current image, they should align with their corresponding points in the previous image.
"""
def timer_func(func): 
    # This function shows the execution time of  
    # the function object passed 
    def wrap_func(*args, **kwargs): 
        t1 = time() 
        result = func(*args, **kwargs) 
        t2 = time() 
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') 
        return result 
    return wrap_func

# https://github.com/creimers/real-time-image-stitching/blob/master/main.py#L106
class Stitcher:
    def __init__(self):
        self.ratio              = 0.85
        self.min_match          = 10 
        self.sift               = cv2.SIFT_create()
        self.smoothing_window   = 800

        self.hp                 = None
        self.wp                 = None
        self.mask_1             = None
        self.mask_2             = None
        self.panorama_1         = None
        self.homography         = None
    def registration(self, img1, img2):
        kp1, des1               = self.sift.detectAndCompute(img1, None)
        kp2, des2               = self.sift.detectAndCompute(img2, None)
        matcher                 = cv2.BFMatcher()
        raw_matches             = matcher.knnMatch(des1, des2, k=2)
        good_points             = []
        good_matches            = []
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3                    = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite("matching.jpg", img3)
        if len(good_points) > self.min_match:
            image1_kp           = np.float32(
                [kp1[i].pt for (_, i) in good_points]
            )
            image2_kp           = np.float32(
                    [kp2[i].pt for (i, _) in good_points]
            )
            H, status           = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
        return H
    def create_mask(self, img1, img2, version):
        height_img1             = img1.shape[0]
        width_img1              = img1.shape[1]
        width_img2              = img2.shape[1]
        height_panorama         = height_img1
        width_panorama          = width_img1 + width_img2
        offset                  = int(self.smoothing_window / 2)
        barrier                 = img1.shape[1] - int(self.smoothing_window/2)
        mask                    = np.zeros((height_panorama, width_panorama))
        if version == "left_image":
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset :barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    @timer_func
    def blending(self, img1, img2):
        H                       = self.registration(img1, img2)
        self.homography         = H
        height_img1             = img1.shape[0]
        width_img1              = img1.shape[1]
        width_img2              = img2.shape[1]
        height_panorama         = height_img1
        width_panorama          = width_img1 + width_img2
        self.hp                 = height_panorama
        self.wp                 = width_panorama

        panorama1               = np.zeros((height_panorama, width_panorama, 3))
        self.panorama_1         = panorama1
        mask1                   = self.create_mask(img1, img2, version="left_image")
        self.mask_1             = mask1
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1               *= mask1
        mask2                   = self.create_mask(img1, img2, version="right_image")
        self.mask_2             = mask2
        panorama2               = cv2.warpPerspective(img2, H, (width_panorama, height_panorama)) * mask2
        result                  = panorama1 + panorama2

        rows, cols              = np.where(result[:, :, 0] != 0)
        min_row, max_row        = min(rows), max(rows) + 1
        min_col, max_col        = min(cols), max(cols) + 1
        final_result            = result[min_row:max_row, min_col:max_col, :]
        print(final_result)
        return final_result

    @timer_func
    def re_blend(self, img1, img2):
        self.panorama_1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        self.panorama_1         *= self.mask_1
        panorama2               = cv2.warpPerspective(img2, self.homography, (self.wp, self.hp)) * self.mask_2
        result                  = self.panorama_1 + panorama2
        rows, cols              = np.where(result[:, :, 0] != 0)
        min_row, max_row        = min(rows), max(rows) + 1
        min_col, max_col        = min(cols), max(cols) + 1
        final_result            = result[min_row:max_row, min_col:max_col, :]
        return final_result

def main(argv1, argv2):
    img1                        = cv2.imread(argv1)
    img2                        = cv2.imread(argv2)
    final                       = Stitcher()
    result                      = final.blending(img1, img2)
    cv2.imwrite("Panorama.jpg", result)
    result                      = final.re_blend(img1, img2)
    cv2.imwrite("NewPanorama.jpg", result)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
