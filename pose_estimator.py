import numpy as np
import cv2
import matplotlib.pyplot as plt

def calibrate_camera(image_1, focal_length, cx, cy):

    points_2d = np.load("./data/vr2d.npy")
    points_3d = np.load("./data/vr3d.npy")

    points_2d = np.squeeze(points_2d)
    points_3d = np.squeeze(points_3d)

    camera_matrix = np.array([[focal_length, 0, cx],
                              [0, focal_length, cy],
                              [0,      0,       1]])

    _, camera_matrix, _, _, _ = cv2.calibrateCamera(
        [points_3d], [points_2d], image_1.shape[::-1], camera_matrix, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS +
                                                                            cv2.CALIB_FIX_PRINCIPAL_POINT +
                                                                            cv2.CALIB_FIX_ASPECT_RATIO)

    return camera_matrix


def find_matches(image_1, image_2):

    #Flann/ORB etc. can be used here as well, just found the sift method accurate and fast enough for this case
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(image_1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image_2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    best_matches = []

    for (m, n) in matches:
        if m.distance < 0.9 * n.distance:
            best_matches.append(m)

    return keypoints_1, keypoints_2, best_matches


def find_translation_and_rotation(keypoints_1, keypoints_2, best_matches, camera_matrix):

    points_1 = np.float32([keypoints_1[m.queryIdx].pt for m in best_matches])
    points_2 = np.float32([keypoints_2[m.trainIdx].pt for m in best_matches])

    E, _ = cv2.findEssentialMat(
        points_1, points_2, camera_matrix, method=cv2.RANSAC, prob=0.999)

    _, rotation_estimation, translation_estimation, _ = cv2.recoverPose(
        E, points_1, points_2, camera_matrix)

    return translation_estimation, rotation_estimation


def camera_pose_estimator(image_1, image_2, focal_length, cx, cy):


    # calibrate the camera with f=100, cx=960, cy=540
    camera_matrix = calibrate_camera(image_1, focal_length, cx, cy)

    keypoints_1, keypoints_2, best_matches = find_matches(image_1, image_2)

    return find_translation_and_rotation(keypoints_1, keypoints_2, best_matches, camera_matrix)

if __name__ == '__main__':

    # First on images 1 and 2
    img1 = cv2.imread("./res/img1.png")
    img2 = cv2.imread("./res/img2.png")
    img3 = cv2.imread("./res/img3.png")

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    focal_length, cx, cy = 100, 960, 540
    translation_12, rotation_12 = camera_pose_estimator(img1, img2, focal_length, cx, cy)
    translation_13, rotation_13 = camera_pose_estimator(img1, img3, focal_length, cx, cy)

    rotation_12 = rotation_12.transpose()
    pos_1 = np.matmul(-rotation_12,translation_12)  
    rotation_13 = rotation_13.transpose()
    pos_2 = np.matmul(-rotation_13,translation_13)
    
    pos_0 = np.zeros_like(pos_1)
    plt.figure()
    plt.xlabel('X')
    plt.ylabel('Y')
    positions = np.array([pos_0, pos_1, pos_2])
    plt.plot(positions[:,0],positions[:,2])

    plt.savefig('./trajectory_plot.png')
    plt.show()


