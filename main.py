import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import warnings
from numpy import RankWarning
warnings.simplefilter('ignore', RankWarning)
import math

import sliding_window as sw
from visualization import create_lane_plot
from image_preprocessing import Bev_Gray_Blurred_Binary_transform
from pixel_to_world import pixel_to_meter
from lane_normal_vector_cal import L_normal_vector_cal, R_normal_vector_cal
from pure_pursuit import lookahead_point_cal, lookahead_point_vec_cal, angle_between_vectors, delta_raidians_cal
from polynomial_fit import R_Polyft_Plotting, L_Polyft_Plotting, p_Polyft_Plotting, nv_p_Polyft_Plotting

# 비디오 캡처
cap = cv2.VideoCapture("test_video_640x480.mp4")

# 플롯 설정
plot_objs     = create_lane_plot()
fig, ax       = plot_objs['fig'], plot_objs['ax']
L_scatter     = plot_objs['L_scatter']
R_scatter     = plot_objs['R_scatter']
p_scatter     = plot_objs['p_scatter']
nv_p_scatter  = plot_objs['nv_p_scatter']
look_scatter  = plot_objs['lookahead_scatter']

# 좌표 변환 파라미터
origin_x, origin_y = (569 - 28) / 2, 480
S_x                = 0.85 / (569 - 28)
S_y                = 0.5  / (247 - 123)

# 슬라이딩 윈도우 파라미터
R_num_windows, L_num_windows           = 25, 6
R_margin, L_margin                     = 60, 150
min_points                             = 10
R_consistency_threshold, L_consistency_threshold = 560, 600

# SlidingWindow 인스턴스 생성
sw_detector = sw.SlidingWindow(
    origin_x, origin_y, S_x, S_y,
    R_num_windows=R_num_windows,
    L_num_windows=L_num_windows,
    R_margin=R_margin,
    L_margin=L_margin,
    min_points=min_points,
    R_consistency_threshold=R_consistency_threshold,
    L_consistency_threshold=L_consistency_threshold,
    x_offset=1.341
)

# Pure Pursuit 파라미터
Ld       = 2.5
car_dir  = [1, 0]

def main():
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) 전처리: BEV→Gray→Blur→Binary
        binary = Bev_Gray_Blurred_Binary_transform(frame)

        # 2) 슬라이딩 윈도우 탐색
        result_frame, R_pts_m, L_pts_m, p_pts, nv_pts = sw_detector.process(
            binary,
            ax, L_scatter, R_scatter,
            p_scatter, nv_p_scatter
        )

        # 3) 경로 및 법선 벡터 경로 다항식 피팅
        if p_pts:
            p_Polyft_Plotting(p_pts, p_scatter, ax)

        nv_poly = None
        if nv_pts:
            degree  = 1 if not R_pts_m else 3
            nv_poly = nv_p_Polyft_Plotting(nv_pts, nv_p_scatter, ax, degree)

        # 4) Pure Pursuit 룩어헤드 & 조향각 계산
        if nv_poly is not None:
            lookahead_list = lookahead_point_cal(0, 0, Ld, nv_poly, nv_pts)
            if lookahead_list:
                for pt in lookahead_list:
                    # 이전 룩어헤드 포인트 제거
                    for line in ax.lines:
                        if line.get_label() == "lookahead_point":
                            line.remove()
                    # 새 룩어헤드 포인트 표시
                    ax.plot(pt[0], pt[1],
                            'o',
                            markerfacecolor='orange',
                            markeredgecolor='black',
                            label="lookahead_point",
                            markersize=10)

                    # 벡터/각도/조향각 계산
                    vec   = lookahead_point_vec_cal(pt[0], pt[1])
                    alpha = angle_between_vectors(car_dir, vec)
                    delta = delta_raidians_cal(alpha)
                    print(f"alpha: {alpha:.3f}, delta: {delta:.3f}")

        # 5) Matplotlib 플롯을 OpenCV 이미지로 변환
        canvas = FigureCanvas(fig)
        canvas.draw()
        plot_img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        plot_img = plot_img.reshape(canvas.get_width_height()[::-1] + (4,))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)

        # 6) 결과 출력
        cv2.imshow("Original Video", frame)
        cv2.imshow("Sliding Window Result", result_frame)
        cv2.imshow("Polynomial Fit Plot", plot_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
