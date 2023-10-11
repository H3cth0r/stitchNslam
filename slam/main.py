import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

class Matcher:
    def __init__(self):
        self.sift       = cv2.SIFT_create()
        self.bf         = BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    def feature_matching(self, img1, img2):

        kp1, des1       = self.sift.detectAndCompute(img1, None)
        kp2, des2       = self.sift.detectAndCompute(img2, None)

        matches         = bf.match(des1, des2)

        # Lowe's ratio test
        good            = [m for m in matches if m.distance < 0.7 * min(m.distance for m in matches)]

        pts1            = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2            = np.float32([kp1[m.trainIdx].pt for m in matches])
        img3            = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return pts1, pts2

class Calibrator:
    def __init__(self):
        self.CHECKBOARD     = (7, 7)
        # stop criteria
        self.criteria       = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    def calibrate(self):
        _3dPoints           = []
        _2dPoints           = []
        
        _3dPointsCoords     = np.zeros((1, self.CHECKBOARD[0] * self.CHECKBOARD[1], 3), np.float32)
        _3dPointsCoords[0, :, :2] = np.mgrid[0:self.CHECKBOARD[0], 0:self.CHECKBOARD[1]].T.reshape(-1, 2)
        
        prev_img_shape      = None

        images              = glob.glob('calibration_imgs/*.jpg')
        for filename in images:
            print("loop")
            image           = cv2.imread(filename)
            grayColor       = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # chess board corners
            # ret : indicates whether the corners ere detected
            ret, corners    = cv2.findChessboardCorners(grayColor,
                                                        self.CHECKBOARD, 
                                                        cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                        cv2.CALIB_CB_FAST_CHECK +
                                                        cv2.CALIB_CB_NORMALIZE_IMAGE
                              )
            # if desired number of corners detectedn then refine
            # pixel coordinates and display them on the images
            if ret == True:
                _3dPoints.append(_3dPointsCoords)
                corners2    = cv2.cornerSubPix(
                                    grayColor, corners, (11, 11), (-1, -1), self.criteria
                              )
                _2dPoints.append(corners2)

                # draw and display corners
                image       = cv2.drawChessboardCorners(image, self.CHECKBOARD, corners2, ret)
            # cv2.imshow("img", image)
            # cv2.waitKey(0)
            plt.imshow(image)
            plt.axis("off")
            plt.show()

        h, w = image.shape[:2]

        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
                                                    _3dPointsCoords, _2dPoints, grayColor.shape[::-1], None, None
                                                  )
        return matrix

if __name__ == "__main__":
    # img1            = cv.imread("")
    # img2            = cv.imread("")
    # matcher         = Matcher()
    # pts1, pts2      = matcher.feature_matching(img1, img2)
    cal             = Calibrator()
    matrix          = cal.calibrate()
    print(matrix)
