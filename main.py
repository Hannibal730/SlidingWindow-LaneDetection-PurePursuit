import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import warnings
from numpy import RankWarning
# polyfit 경고 무시
warnings.simplefilter('ignore', RankWarning)
import math
from plot_setup import create_lane_plot
from preprocessing import Bev_Gray_Blurred_Binary_transform, histogram_argmax
from coordinate_transform import pixel_to_meter
from normal_vector_cal import L_normal_vector_cal, R_normal_vector_cal
from pure_pursuit import lookahead_point_cal, lookahead_point_vec_cal, angle_between_vectors, delta_raidians_cal
from poly_fit import R_Polyft_Plotting, L_Polyft_Plotting, p_Polyft_Plotting, nv_p_Polyft_Plotting

cap = cv2.VideoCapture("test_video_640x480.mp4")

# 플롯 설정
plot_objs = create_lane_plot()
fig = plot_objs['fig']
ax = plot_objs['ax']
L_scatter = plot_objs['L_scatter']
R_scatter = plot_objs['R_scatter']
p_scatter = plot_objs['p_scatter']
nv_p_scatter = plot_objs['nv_p_scatter']
lookahead_scatter = plot_objs['lookahead_scatter']

# 좌표 변환 설정
origin_x, origin_y = (569 - 28) / 2, 480
S_x = 0.85 / (569 - 28)
S_y = 0.5 / (247 - 123)

# 슬라이딩 윈도우 파라미터
R_num_windows, L_num_windows = 25, 6
R_margin, L_margin = 60, 150
min_points = 10
R_consistency_threshold, L_consistency_threshold = 560, 600
L_win_x_prev, R_win_x_prev = None, None

# 윈도우 인포 함수
def window_info(frame, num_w, win_idx, win_x_curr, margin):
    h = int(frame.shape[0] / num_w)
    y_low = frame.shape[0] - (win_idx + 1) * h
    y_high = frame.shape[0] - win_idx * h
    x_low = max(0, win_x_curr - margin)
    x_high = min(frame.shape[1], win_x_curr + margin)
    idxs = np.column_stack(np.where(frame[y_low:y_high, x_low:x_high] > 0))
    if len(idxs) == 0:
        return h, y_low, y_high, x_low, x_high, idxs, None, None
    x_mean = int(np.mean([x_low + i[1] for i in idxs]))
    y_mean = int(np.mean([y_low + i[0] for i in idxs]))
    return h, y_low, y_high, x_low, x_high, idxs, x_mean, y_mean

# pure pursuit 관련
Ld = 2.5
car_dir = [1, 0]

# 메인 함수
def main():
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pre = Bev_Gray_Blurred_Binary_transform(frame)
        hist, Lx_arg, Rx_arg, L_thresh, R_thresh = histogram_argmax(pre)
        result = np.dstack((pre,)*3)

        global L_win_x_prev, R_win_x_prev
        L_win_x_prev = Lx_arg if L_win_x_prev is None else L_win_x_prev
        R_win_x_prev = Rx_arg if R_win_x_prev is None else R_win_x_prev
        L_win_x = Lx_arg if Lx_arg >= L_thresh else L_win_x_prev
        R_win_x = Rx_arg if Rx_arg >= R_thresh else R_win_x_prev

        R_pts_m, L_pts_m, p_pts, nv_pts = [], [], [], []

        # 오른쪽 윈도우
        for i in range(R_num_windows):
            if i == 0 and R_win_x < 460:
                R_win_x = 460
            _, y_low, y_high, x_low, x_high, idxs, xm, ym = window_info(pre, R_num_windows, i, R_win_x, R_margin)
            if len(idxs) > min_points:
                R_win_x = xm if abs(xm - R_win_x_prev) < R_consistency_threshold else R_win_x_prev
                cv2.rectangle(result, (x_low, y_low), (x_high, y_high), (0,0,255), 2)
                cv2.circle(result, (xm, ym), 5, (0,0,255), -1)
                mx, my = pixel_to_meter(xm, ym, origin_x, origin_y, S_x, S_y)
                R_pts_m.append((mx+1.341, my))
                p_pts.append((mx+1.341, my+0.425))
                poly_R = R_Polyft_Plotting(R_pts_m, R_scatter, ax)
                nv = R_normal_vector_cal(mx+1.341, poly_R)
                nv_pts.append((mx+1.341, my) + nv)
            else:
                break
        R_win_x_prev = R_win_x

        if R_pts_m:
            R_Polyft_Plotting(R_pts_m, R_scatter, ax)

        # 왼쪽 윈도우
        for i in range(L_num_windows):
            if i == 0 and L_win_x > 20:
                L_win_x = 20
            _, y_low, y_high, x_low, x_high, idxs, xm, ym = window_info(pre, L_num_windows, i, L_win_x, L_margin)
            if len(idxs) > min_points:
                if pre[ym, xm] == 255:
                    L_win_x = xm if abs(xm - L_win_x_prev) < L_consistency_threshold else L_win_x_prev
                    cv2.rectangle(result, (x_low, y_low), (x_high, y_high), (255,0,0), 2)
                    cv2.circle(result, (xm, ym), 7, (255,0,0), -1)
                    mx, my = pixel_to_meter(xm, ym, origin_x, origin_y, S_x, S_y)
                    L_pts_m.append((mx+1.341, my))
                else:
                    left_idxs = np.column_stack(np.where(pre[y_low:y_high, :xm] > 0))
                    if len(left_idxs) > min_points:
                        lx = int(np.mean([x_low + i[1] for i in left_idxs]))
                        ly = int(np.mean([y_low + i[0] for i in left_idxs]))
                        cv2.rectangle(result, (x_low,y_low),(x_high,y_high),(0,255,0),2)
                        cv2.circle(result,(lx,ly),7,(0,255,0),-1)
                        mx, my = pixel_to_meter(lx, ly, origin_x, origin_y, S_x, S_y)
                        L_pts_m.append((mx+1.341, my))
            else:
                break
        L_win_x_prev = L_win_x
        if L_pts_m:
            L_Polyft_Plotting(L_pts_m, L_scatter, ax)

        # 경로 피팅
        if p_pts:
            p_Polyft_Plotting(p_pts, p_scatter, ax)
        # 법선 벡터 피팅 후 함수 저장
        nv_poly = None
        if nv_pts:
            degree = 1 if not R_pts_m else 3
            nv_poly = nv_p_Polyft_Plotting(nv_pts, nv_p_scatter, ax, degree)

        # 룩어헤드 포인트 계산
        if nv_poly is not None:
            lookahead_pts = lookahead_point_cal(0, 0, Ld, nv_poly, nv_pts)
            if lookahead_pts:
                for pt in lookahead_pts:
                    for line in ax.lines:
                        if line.get_label() == "lookahead_point":
                            line.remove()
                    ax.plot(pt[0], pt[1], 'o', markerfacecolor='orange', markeredgecolor='black', label="lookahead_point", markersize=10)
                    vec = lookahead_point_vec_cal(pt[0], pt[1])
                    alpha = angle_between_vectors(car_dir, vec)
                    delta = delta_raidians_cal(alpha)
                    print(f"alpha: {alpha}, delta: {delta}, deg: {math.degrees(delta)}")

        # 플롯 렌더링
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        img = img.reshape(canvas.get_width_height()[::-1] + (4,))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        cv2.imshow("Original Video", frame)
        cv2.imshow("Sliding Window Result", result)
        cv2.imshow("Polynomial Fit Plot", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
    cap.release()
    cv2.destroyAllWindows()
