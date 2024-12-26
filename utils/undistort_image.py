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

    dist_coeffs = np.array([k1, k2, p1, p2, k3])  
    Args:
        cam_in: camera intrinsic parameters, [fx, fy, cx, cy]
        cam_dist_coeffs: camera distortion parameters, [k1, k2, p1, p2, k3]
    
    """

    h, w = img_shape
    camera_matrix = np.array(cam_in)
    dist_coeffs = np.array(cam_dist_coeffs)

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), 5)
    return [map1, map2, roi]

def undistort_image(img, map1, map2, roi):

    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    x, y, w, h = roi
    undistorted_img_crop = undistorted_img[y:y+h, x:x+w]
    return undistorted_img, undistorted_img_crop


if __name__ == '__main__':
    img_path = '/horizon-bucket/saturn_v_release/005_perception_static/03_hde_data/driving_auto/v163/FSD_Site_DZ878_20240622/Site_114_32953_22_67792_0/DZ878_20240622_020406/camera_front/1718993635400.jpg'
    test_img = cv2.imread(img_path)
    map_roi = get_undistort_map(test_img.shape[:2])
    undistorted_img, undistorted_img_crop = undistort_image(test_img, map_roi[0], map_roi[1], map_roi[2])
    cv2.imwrite('test_crop.jpg', undistorted_img_crop)
    cv2.imwrite('test.jpg', undistorted_img)

    cv2.imwrite('test_ori.jpg', test_img)