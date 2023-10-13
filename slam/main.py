import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

class Matcher:
    def __init__(self):
        self.sift       = cv2.ORB_create()
        self.bf         = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    def feature_matching(self, img1, img2):
        kp1, des1       = self.sift.detectAndCompute(img1, None)
        kp2, des2       = self.sift.detectAndCompute(img2, None)

        matches         = self.bf.match(des1, des2)

        # Lowe's ratio test
        # matches = [m for m in matches if m.distance < 0.7 * min(m.distance for m in matches)]
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:int(len(matches) * 0.8)]

        pts1            = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,2)
        pts2            = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,2)
        return pts1, pts2
class MotionEstimation:
    def __init__(self, file_name):
        self.K          = np.load(file_name)
    def estimate(self, pts1_t, pts2_t):
        F, mask             = cv2.findFundamentalMat(pts1_t, pts2_t, cv2.FM_LMEDS)
        E                   = np.dot(self.K.T, np.dot(F, self.K))
        retval, R, t, mask  = cv2.recoverPose(E, pts1_t, pts2_t, self.K)
        return R, t 
def plot_transformation(camera_positions_t):
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")

    # Plot the camera positions connected with a line

    ax.scatter(*camera_positions_t[0], c='g', marker='o', label='Initial Point')
    ax.scatter(*camera_positions_t[-1], c='r', marker='o', label='Final Point')

    X = [pos[0] for pos in camera_positions_t]
    Y = [pos[1] for pos in camera_positions_t]
    Z = [pos[2] for pos in camera_positions_t]

    # Plot lines connecting the camera positions
    ax.plot(X, Y, Z, 'b-', label='Camera Path')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    plt.show()
def plot_points_with_color(feature_3d_points):
    # Extract Z coordinates (depth)
    Z = np.array([point[2] for point in feature_3d_points])

    # Normalize the Z coordinates to [0, 1] for color mapping
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.viridis(norm(Z))

    # Create a 3D scatter plot with color scaling based on Z-coordinate
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([point[0] for point in feature_3d_points],
               [point[1] for point in feature_3d_points],
               [point[2] for point in feature_3d_points],
               c=colors, marker='o', label='3D Feature Points', alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Create a colorbar to show the color scale
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Depth (Z-coordinate)')

    plt.legend()
    plt.show()
if __name__ == "__main__":
    matcher         = Matcher()
    me              = MotionEstimation("./camera_calibration_matrix.npy")

    video_capture   = cv2.VideoCapture("./test_videos/test_1.mp4")
    ret, prev_frame = video_capture.read()
    camera_positions= []
    camera_positions.append(np.array([0, 0, 0]))

    feature_3d_points=[]

    while True:
        ret, frame  = video_capture.read()
        if not ret:
            break
        pts1, pts2  = matcher.feature_matching(prev_frame, frame)
        R, t        = me.estimate(pts1, pts2)
        # new_frame   = camera_positions[-1] + np.dot(R, t)
        new_frame = camera_positions[-1] + np.dot(-R.T, -t).reshape(-1)
        # new_frame = camera_positions[-1] + np.dot(R.T, t).reshape(-1)
        camera_positions.append(new_frame)
        print(len(camera_positions))

        frame_with_points = cv2.drawKeypoints(frame, [cv2.KeyPoint(pt[0], pt[1], 5) for pt in pts2], None, color=(0, 255, 0), flags=0)

        pts3D = cv2.triangulatePoints(np.eye(3, 4), np.column_stack((R, t)), pts1.T, pts2.T).T
        pts3D /= pts3D[:, 3][:, np.newaxis]
        feature_3d_points.extend(pts3D[:, :3])

        for pt in pts2:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
        cv2.imshow('Frame with Feature Points', frame_with_points)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

    plot_points_with_color(feature_3d_points)
    plot_transformation(np.vstack(camera_positions))
