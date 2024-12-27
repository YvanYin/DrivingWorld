import cv2
import numpy as np


def get_undistort_map(
    img_shape, 
    cam_in=[[1905.89892578125, 0.0, 1918.4837646484375], [0.0, 1905.89892578125, 1075.7330322265625], [0.0, 0.0, 1.0]], 
    cam_dist_coeffs=[0.8017038106918335, 0.10657747834920883, 1.5339870742536732e-06, -7.50786193748354e-06, 0.0010572359897196293, 1.1659871339797974, 0.30344289541244507, 0.011946137063205242]
    ):
    """
    Rectify camera distortions.
    Camera intrinsic parameters and distortions examples are as follows.    
    camera_matrix = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]])

    dist_coeffs = np.array([k1, k2, p1, p2, k3])  # Depending on the distortion model used, this array might have fewer or more values.
    Args:
        cam_in: camera intrinsic parameters, [fx, fy, cx, cy]
        cam_dist_coeffs: camera distortion parameters, [k1, k2, p1, p2, k3]
    
    """
    # Get the dimensions of the image
    h, w = img_shape
    camera_matrix = np.array(cam_in)
    dist_coeffs = np.array(cam_dist_coeffs)

    # Get the optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # # Undistort the image
    # undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Map coordinates from new camera matrix to distorted image using initUndistortRectifyMap which inversely remaps for distortion
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), 5)
    return [map1, map2, roi]

def undistort_image(img, map1, map2, roi):
    # Reversely apply maps to simulate distortion
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    x, y, w, h = roi
    undistorted_img_crop = undistorted_img[y:y+h, x:x+w]
    return undistorted_img, undistorted_img_crop
