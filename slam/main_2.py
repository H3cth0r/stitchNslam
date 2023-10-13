import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_calibration_matrices():
    return
class VisualOdometry:
    def __init__(self, K_t, P_t):
        self.K              = K_t
        self.P              = P_t
        self.orb            = cv2.ORB_create(3000)
        FLANN_INDEX_LSH     =   6
        index_params        = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params       = dict(checks=50)
        self.flann          = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
    def form_transform(self, R, t):
        T                   = np.eye(4, dtype=np.float64)
        T[:3, :3]           = R
        T[:3,  3]           = t
        return T
    def get_matches(self, prev_frame, curr_frame):
        kp1, des1           = self.orb.detectAndCompute(prev_frame, None)
        kp2, des2           = self.orb.detectAndCompute(curr_frame, None)
        
        matches             = self.flann.knnMatch(des1, des2, k=2)

        good                = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        draw_params         = dict(matchesThickness=2, matchColor=-1, singlePointColor=None, matchesMask=None, flags=2)
        img3                = cv2.drawMatches(curr_frame, kp1, prev_frame,kp2, good ,None,**draw_params)
        cv2.imshow("image", img3)
        cv2.waitKey(200)

        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2
    def get_pose(self, q1, q2):
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)
        R, t = self.decomp_essential_matrix(E, q1, q2)
        transformation_matrix = self.form_transform(R, np.squeeze(t))
        return transformation_matrix
    def sum_z_cal_relative_scale(self, R, t):
        T                   = self.form_transform(R, t)
        P                   = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)
        hom_Q1              = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
        hom_Q2              = np.matmul(T, hom_Q1)

        uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

        sum_of_pos_z_Q1     = sum(uhom_Q1[2, :] > 0)
        sum_of_pos_z_Q2     = sum(uhom_Q2[2, :] > 0)
        relative_scale      = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                        np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
        return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

    def decomp_essential_matrix(self, E, q1, q2):
        R1, R2, t           = cv2.decomposeEssentialMat(E)
        t                   = np.squeeze(t)

        pairs               = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        z_sums              = []
        relative_scales     = []
        for R, t in pairs:
            z_sum, scale    = self.sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]
def load_poses(filepath):
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

def plot_camera_trajectory(transformation_matrices):
    # Initialize the camera positions with the identity matrix (initial position)
    camera_positions = [np.eye(4)]

    for matrix in transformation_matrices:
        # Accumulate the camera poses
        new_position = np.dot(camera_positions[-1], matrix)
        camera_positions.append(new_position)

    # Extract X, Y, and Z coordinates from the camera positions
    X = [pose[0, 3] for pose in camera_positions]
    Y = [pose[1, 3] for pose in camera_positions]
    Z = [pose[2, 3] for pose in camera_positions]

    # Create a 3D plot of the camera trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X, Y, Z, label='Camera Trajectory', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory')

    plt.legend()
    plt.show()
if __name__ == "__main__":
    K                   = np.load("./camera_calibration_matrix.npy")
    P                   = np.load("./camera_projection_matrix.npy")
    gt_poses            = load_poses("./init_poses.txt")
    visualOdometry      = VisualOdometry(K, P)
    video_capture       = cv2.VideoCapture("./test_videos/test_2.mp4")
    ret, prev_frame     = video_capture.read()
    transformation_matrices = [] 
    while True:
        ret, cur_frame  = video_capture.read()
        if not ret:
            break
        q1, q2          = visualOdometry.get_matches(prev_frame, cur_frame)
        transf          = visualOdometry.get_pose(q1, q2)
        prev_frame      = cur_frame
        print("="*60)
        print(transf)
        transformation_matrices.append(transf)
    plot_camera_trajectory(transformation_matrices)


