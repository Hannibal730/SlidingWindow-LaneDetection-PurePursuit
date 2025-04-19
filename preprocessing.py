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


def histogram_argmax(frame):
    """
    하단부 흰 픽셀 히스토그램을 계산하여
    좌/우 차선 초기 x 위치와 threshold 값을 반환.
    
    Returns:
      histogram         : 1D 히스토그램 배열
      L_x_hist_argmax  : 왼쪽 최대값 x좌표
      R_x_hist_argmax  : 오른쪽 최대값 x좌표
      L_x_threshold    : 왼쪽 유효 임계 x
      R_x_threshold    : 오른쪽 유효 임계 x
    """
    # 하단 2/3 영역 합산
    h = frame.shape[0]
    hist = np.sum(frame[int(1.15 * h / 3):, :], axis=0)
    mid = hist.shape[0] // 2

    L_x = np.argmax(hist[:mid])
    R_x = np.argmax(hist[mid:]) + mid

    # 유효 범위 임계치
    L_thr = int(1   * hist.shape[0] / 3)
    R_thr = int(1.6 * hist.shape[0] / 3)

    return hist, L_x, R_x, L_thr, R_thr
