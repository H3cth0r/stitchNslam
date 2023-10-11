import cv2
import numpy as np

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
if __name__ == "__main__":
    print("lol")
    # img1            = cv.imread("")
    # img2            = cv.imread("")
    # matcher         = Matcher()
    # pts1, pts2      = matcher.feature_matching(img1, img2)
