import cv2 as cv
import numpy as np

# === 설정 ===
video_file = 'video.mp4'
board_pattern = (7, 7)             # 체스보드 코너 수 (가로, 세로)
board_cellsize = 0.025             # 체스보드 셀 크기 (단위: m)
select_all = False                 # 자동 저장 or 수동 선택

# === 프레임 선택 함수 ===
def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10):
    video = cv.VideoCapture(video_file)
    img_select = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        found, corners = cv.findChessboardCorners(frame, board_pattern)
        disp = frame.copy()
        cv.drawChessboardCorners(disp, board_pattern, corners, found)
        cv.imshow('Select Frame (SPACE: select, ESC: done)', disp)

        key = cv.waitKey(wait_msec) & 0xFF
        if found and (select_all or key == ord(' ')):
            img_select.append(frame.copy())
            print(f"Selected frame #{len(img_select)}")
        if key == 27:  # ESC
            break

    video.release()
    cv.destroyAllWindows()
    return img_select

# === 카메라 보정 함수 ===
def calib_camera_from_chessboard(images, board_pattern, board_cellsize):
    img_points = []

    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        success, pts = cv.findChessboardCorners(gray, board_pattern)
        if success:
            img_points.append(pts)

    assert len(img_points) > 0, "No complete chessboard patterns found!"

    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points)

    gray_shape = cv.cvtColor(images[0], cv.COLOR_BGR2GRAY).shape[::-1]
    ret, K, dist_coeff, rvecs, tvecs = cv.calibrateCamera(
        obj_points, img_points, gray_shape, None, None
    )

    return K, dist_coeff, ret

# === 왜곡 보정 실행 함수 ===
def run_undistort_video(video_file, K, dist_coeff):
    video = cv.VideoCapture(video_file)
    if not video.isOpened():
        print("[ERROR] Cannot open video.")
        return

    map1, map2 = None, None
    show_rectify = True

    while True:
        valid, img = video.read()
        if not valid:
            print("[INFO] Video playback finished.")
            break

        if show_rectify:
            if map1 is None or map2 is None:
                h, w = img.shape[:2]
                map1, map2 = cv.initUndistortRectifyMap(
                    K, dist_coeff, None, K, (w, h), cv.CV_32FC1
                )
            undistorted = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
            info = "Rectified"
        else:
            undistorted = img
            info = "Original"

        cv.putText(undistorted, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
        cv.imshow("Camera Undistortion", undistorted)

        key = cv.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            show_rectify = not show_rectify  # 원본/보정 전환

    video.release()
    cv.destroyAllWindows()

# === 메인 ===
if __name__ == '__main__':
    print("[STEP 1] Selecting calibration frames from video...")
    selected_images = select_img_from_video(video_file, board_pattern, select_all)

    if not selected_images:
        print("[ERROR] No calibration images selected.")
        exit()

    print("[STEP 2] Running camera calibration...")
    K, dist_coeff, error = calib_camera_from_chessboard(selected_images, board_pattern, board_cellsize)

    print("\n[RESULT] Calibration completed.")
    print("Reprojection Error:", error)
    print("Camera Matrix (K):\n", K)
    print("Distortion Coefficients:\n", dist_coeff.ravel())

    print("\n[STEP 3] Running undistortion on video...")
    run_undistort_video(video_file, K, dist_coeff)
