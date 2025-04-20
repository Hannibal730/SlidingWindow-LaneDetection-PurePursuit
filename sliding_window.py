import cv2
import numpy as np
from lane_histogram import histogram_argmax
from pixel_to_world import pixel_to_meter
from polynomial_fit import R_Polyft_Plotting, L_Polyft_Plotting
from lane_normal_vector_cal import R_normal_vector_cal, L_normal_vector_cal


class SlidingWindow:
    """
    슬라이딩 윈도우 기반 차선 픽셀 수집, 좌표 변환, 그리고 초기 다항식 피팅을 수행합니다.
    메인 루프에서 프레임별로 호출하여 결과 프레임과 포인트 리스트를 반환합니다.
    """
    def __init__(self,
                 origin_x, origin_y, S_x, S_y,
                 R_num_windows=25, L_num_windows=6,
                 R_margin=60, L_margin=150,
                 min_points=10,
                 R_consistency_threshold=560,
                 L_consistency_threshold=600,
                 x_offset=1.341):
        # 월드 변환 파라미터
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.S_x = S_x
        self.S_y = S_y
        self.x_offset = x_offset

        # 슬라이딩 윈도우 설정
        self.R_num_windows = R_num_windows
        self.L_num_windows = L_num_windows
        self.R_margin = R_margin
        self.L_margin = L_margin
        self.min_points = min_points
        self.R_consistency_threshold = R_consistency_threshold
        self.L_consistency_threshold = L_consistency_threshold

        # 이전 프레임 윈도우 x 위치
        self.L_win_x_prev = None
        self.R_win_x_prev = None

    def process(self,
                binary_frame,
                ax, L_scatter, R_scatter,
                p_scatter, nv_p_scatter):
        """
        binary_frame: 전처리된 0/255 바이너리 영상 (np.ndarray)
        ax, *_scatter: Matplotlib 핸들(Scatter, Axes)

        반환:
          result_frame (컬러 결과 영상),
          R_pts_m, L_pts_m, p_pts, nv_pts (각 포인트 리스트)
        """
        # 히스토그램으로 초기 X 추정
        hist, Lx_arg, Rx_arg, L_thr, R_thr = histogram_argmax(binary_frame)
        result_frame = np.dstack((binary_frame,)*3)

        # 이전 값 초기화 또는 유지
        if self.L_win_x_prev is None:  self.L_win_x_prev = Lx_arg
        if self.R_win_x_prev is None:  self.R_win_x_prev = Rx_arg
        L_win_x = Lx_arg if Lx_arg >= L_thr else self.L_win_x_prev
        R_win_x = Rx_arg if Rx_arg >= R_thr else self.R_win_x_prev

        R_pts_m, L_pts_m, p_pts, nv_pts = [], [], [], []

        # 오른쪽 윈도우 반복
        for i in range(self.R_num_windows):
            if i == 0 and R_win_x < 460:
                R_win_x = 460
            # 윈도우 정보 계산
            wh, y_low, y_high, x_low, x_high, idxs, xm, ym = \
                window_info(binary_frame, self.R_num_windows, i, R_win_x, self.R_margin)
            if len(idxs) < self.min_points:
                break
            # 일관성 체크
            R_win_x = xm if abs(xm - self.R_win_x_prev) < self.R_consistency_threshold else self.R_win_x_prev
            # 시각화
            cv2.rectangle(result_frame, (x_low, y_low), (x_high, y_high), (0,0,255), 2)
            cv2.circle(result_frame, (xm, ym), 5, (0,0,255), -1)
            # 월드 좌표 변환
            mx, my = pixel_to_meter(xm, ym, self.origin_x, self.origin_y, self.S_x, self.S_y)
            # 리스트 저장
            R_pts_m.append((mx + self.x_offset, my))
            p_pts.append((mx + self.x_offset, my + 0.425))
            # 피팅 및 법선
            poly_R = R_Polyft_Plotting(R_pts_m, R_scatter, ax)
            nv = R_normal_vector_cal(mx + self.x_offset, poly_R)
            nv_pts.append((mx + self.x_offset, my) + nv)
        self.R_win_x_prev = R_win_x
        if R_pts_m:
            R_Polyft_Plotting(R_pts_m, R_scatter, ax)

        # 왼쪽 윈도우 반복
        for i in range(self.L_num_windows):
            if i == 0 and L_win_x > 20:
                L_win_x = 20
            wh, y_low, y_high, x_low, x_high, idxs, xm, ym = \
                window_info(binary_frame, self.L_num_windows, i, L_win_x, self.L_margin)
            if len(idxs) < self.min_points:
                break
            # 데이터 유무에 따라 색상 구분
            if binary_frame[ym, xm] == 255:
                L_win_x = xm if abs(xm - self.L_win_x_prev) < self.L_consistency_threshold else self.L_win_x_prev
                cv2.rectangle(result_frame, (x_low, y_low), (x_high, y_high), (255,0,0), 2)
                cv2.circle(result_frame, (xm, ym), 7, (255,0,0), -1)
                mx, my = pixel_to_meter(xm, ym, self.origin_x, self.origin_y, self.S_x, self.S_y)
                L_pts_m.append((mx + self.x_offset, my))
            else:
                # 왼쪽 작은 윈도우
                left_idxs = np.column_stack(np.where(binary_frame[y_low:y_high, :xm] > 0))
                if len(left_idxs) < self.min_points:
                    break
                lx = int(np.mean([x_low + i2[1] for i2 in left_idxs]))
                ly = int(np.mean([y_low + i2[0] for i2 in left_idxs]))
                cv2.rectangle(result_frame, (x_low, y_low), (x_high, y_high), (0,255,0), 2)
                cv2.circle(result_frame, (lx, ly), 7, (0,255,0), -1)
                mx, my = pixel_to_meter(lx, ly, self.origin_x, self.origin_y, self.S_x, self.S_y)
                L_pts_m.append((mx + self.x_offset, my))
        self.L_win_x_prev = L_win_x
        if L_pts_m:
            L_Polyft_Plotting(L_pts_m, L_scatter, ax)

        return result_frame, R_pts_m, L_pts_m, p_pts, nv_pts


def window_info(frame, num_w, win_idx, win_x_curr, margin):
    """
    한 개 윈도우의 위치와 흰 픽셀 인덱스, 평균 좌표 반환
    """
    window_h = int(frame.shape[0] / num_w)
    y_low  = frame.shape[0] - (win_idx + 1) * window_h
    y_high = frame.shape[0] - win_idx * window_h
    x_low  = max(0, win_x_curr - margin)
    x_high = min(frame.shape[1], win_x_curr + margin)
    idxs   = np.column_stack(np.where(frame[y_low:y_high, x_low:x_high] > 0))
    if len(idxs) == 0:
        return window_h, y_low, y_high, x_low, x_high, idxs, None, None
    x_mean = int(np.mean([x_low + p[1] for p in idxs]))
    y_mean = int(np.mean([y_low + p[0] for p in idxs]))
    return window_h, y_low, y_high, x_low, x_high, idxs, x_mean, y_mean
