import cv2
import numpy as np

def Bev_Gray_Blurred_Binary_transform(frame):
    """
    Bird’s Eye View → 그레이스케일 → 가우시안 블러 → 이진화
    입력: BGR frame (numpy array)
    출력: binary_frame (0/255)
    """
    height, width = frame.shape[:2]

    # 투시 변환용 원본/목적지 4점 지정
    src_pts = np.float32([[10, 450],
                          [540, 430],
                          [435, 240],
                          [190, 240]])
    dst_pts = np.float32([[0, height],
                          [width, height],
                          [width, 0],
                          [0, 0]])

    # BEV 변환
    bev_matrix  = cv2.getPerspectiveTransform(src_pts, dst_pts)
    bev_frame   = cv2.warpPerspective(frame, bev_matrix, (width, height))

    # 그레이 → 블러 → 이진화
    gray_frame    = cv2.cvtColor(bev_frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (19, 19), 0)
    _, binary_frame = cv2.threshold(blurred_frame, 225, 255, cv2.THRESH_BINARY)

    return binary_frame



