import numpy as np


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