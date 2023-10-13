import cv2
import numpy as np
import glob

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
        example_image       = None
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
                image           = cv2.drawChessboardCorners(image, self.CHECKBOARD, corners2, ret)
                example_image   = image
            # cv2.imshow("img", image)
            # cv2.waitKey(1000)

        cv2.imshow("img", example_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        h, w = image.shape[:2]

        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
                                                    _3dPoints, _2dPoints, grayColor.shape[::-1], None, None
                                                  )
        P = np.zeros((3, 4))
        P[:3, :3] = matrix
        P[:, 3] = np.array([0, 0, 0])
        return matrix, P

if __name__ == "__main__":
    cal             = Calibrator()
    matrix, P       = cal.calibrate()
    np.save("camera_calibration_matrix.npy", matrix)
    np.save("camera_projection_matrix.npy", P)
    print(matrix)
    print("="*20)
    print(P)
