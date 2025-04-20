import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import warnings
from numpy import RankWarning
import math
from plot_setup import create_lane_plot
from preprocessing import Bev_Gray_Blurred_Binary_transform, histogram_argmax
from coordinate_transform import pixel_to_meter
from normal_vector_cal import L_normal_vector_cal, R_normal_vector_cal
from pure_pursuit import lookahead_point_cal, lookahead_point_vec_cal, angle_between_vectors, delta_raidians_cal
from poly_fit import R_Polyft_Plotting, L_Polyft_Plotting, p_Polyft_Plotting, nv_p_Polyft_Plotting

cap = cv2.VideoCapture("test_video_640x480.mp4")



#########################################################################################
# 플롯 세팅
###############################################################q##########################
plot_objs = create_lane_plot()
fig = plot_objs['fig']
ax  = plot_objs['ax']
L_scatter    = plot_objs['L_scatter']
R_scatter    = plot_objs['R_scatter']
p_scatter    = plot_objs['p_scatter']
nv_p_scatter = plot_objs['nv_p_scatter']
lookahead_scatter = plot_objs['lookahead_scatter']




#########################################################################################
# 픽셀 좌표계에서 차량의 위치(541/2, 433)가 미터 좌표계 (0, 0)에 대응하게 설정
#########################################################################################
origin_x, origin_y = (569-28) / 2, 480  # 미터 좌표계에서 원점이 될 픽셀 좌표.

actuaL_lane_width = 0.85  # 차선 간격의 실제 길이 (단위: m)
lane_pixeL_width = 569-28    # 차선 간격의 픽셀 길이
S_x = actuaL_lane_width / lane_pixeL_width  # Y축 변환 비율

actuaL_lane_length = 0.5  # 차선 길이의 실제 길이 (단위: m)
lane_pixeL_length = 247-123    # 차선 길이의 픽셀 길이
S_y = actuaL_lane_length / lane_pixeL_length  # Y축 변환 비율





#########################################################################################
# 윈도우 관련 파라미터, 함수 설정
#########################################################################################
R_num_windows = 25 
L_num_windows = 6  

R_margin = 60  
L_margin = 150  

min_points = 10  

R_consistency_threshold = 560 # 프레임 간 차선 위치 변경 허용 범위
L_consistency_threshold = 600  

L_win_x_prev = None # 첫번째 프레임의 차선 위치 초기화
R_win_x_prev = None

def window_info(frame, num_windows, window, win_x_current, margin):
    window_height = int(frame.shape[0] / num_windows) 
    win_y_low = preprocessed_frame.shape[0] - (window + 1) * window_height
    win_y_high = preprocessed_frame.shape[0] - window * window_height
    win_x_low = max(0, win_x_current - margin)
    win_x_high = min(preprocessed_frame.shape[1], win_x_current + margin)
    win_indices = np.column_stack(np.where(frame[win_y_low:win_y_high, win_x_low:win_x_high] > 0))

    if len(win_indices) == 0: # 흰 픽셀이 없을 경우
        return window_height, win_y_low, win_y_high, win_x_low, win_x_high, win_indices, None, None
    win_x_mean = int(np.mean([win_x_low + idx[1] for idx in win_indices])) # 평균 (윈도우의 왼쪽 경계의 x축 좌표 + 윈도우 내에서 탐지되는 데이터의 '상대적인' x축 좌표. 이때 상대적이라는 것은 윈도우 왼쪽 경계의 x축 좌표를 0으로 바라보고 구했다는 의미이다.)
    win_y_mean = int(np.mean([win_y_low + idx[0] for idx in win_indices])) # 평균 (윈도우 아래 경계의 y축 좌표+ 윈도우 내에서 탐지되는 데이터의 '상대적인' y축 좌표)
    return window_height, win_y_low, win_y_high, win_x_low, win_x_high, win_indices, win_x_mean, win_y_mean







#########################################################################################
# 퓨어퍼슛 관련 파라미터
#########################################################################################
Ld = 2.5
car_direcrtion_vec = [1, 0]



#########################################################################################
# 윈도우 제작 후 프레임 반복 시작
#########################################################################################
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()
    
while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2] # 높이 너비 채널 중 높이 너비만


    # 프레임 처리
    preprocessed_frame = Bev_Gray_Blurred_Binary_transform(frame) # 영상 전처리
    histogram, L_x_hist_argmax, R_x_hist_argmax, L_x_threshold, R_x_threshold = histogram_argmax (preprocessed_frame) # 히스토그램 계산
    result_frame = np.dstack((preprocessed_frame, preprocessed_frame, preprocessed_frame)) # 윈도우와 데이터 포인트를 컬러로 쌓을 거라서 컬러 채널화

    # 초기화
    L_win_x_prev = L_x_hist_argmax if L_win_x_prev is None else L_win_x_prev # 첫 프레임일 때는 이전 프레임값은 없으니 hist_argmax 값을 배정함.    
    R_win_x_prev = R_x_hist_argmax if R_win_x_prev is None else R_win_x_prev
    L_win_x_current = L_x_hist_argmax if L_x_hist_argmax >= L_x_threshold else L_win_x_prev
    R_win_x_current = R_x_hist_argmax if R_x_hist_argmax >= R_x_threshold else L_win_x_prev
    
    
    
    nv_p_list = [] # 법선벡터의 경로함수
    p_list = [] # 경로함수의 좌표 저장 리스트
    R_points_meter_list = [] # 오른쪽 윈도우 미터 단위 좌표 저장
    L_points_meter_list = []

    # 오른쪽 윈도우
    for window in range(R_num_windows):
        if window == 0:
            if R_win_x_current < 460:
                R_win_x_current = 460
        R_window_height, R_win_y_low, R_win_y_high, R_win_x_low, R_win_x_high, R_win_indices, R_win_x_mean, R_win_y_mean = window_info(preprocessed_frame, R_num_windows, window, R_win_x_current, R_margin)
        # 윈도우에서 데이터가 탐지된다면
        if len(R_win_indices) > min_points:
            R_win_x_current = R_win_x_mean if abs(R_win_x_mean - R_win_x_prev) < R_consistency_threshold else R_win_x_prev # 새로 구한 x좌표와 기존 x 좌표 사이의 거리가 쓰레숄드를 넘지 않으면 새로 구한 x좌표를 R_win_x_current로 갱신. 넘으면 기존 x좌표 유지
            cv2.rectangle(result_frame, (R_win_x_low, R_win_y_low), (R_win_x_high, R_win_y_high), (0, 0, 255), 2) # 빨간색 윈도우랑 데이터 포인트
            cv2.circle(result_frame, (R_win_x_mean, R_win_y_mean), 5, (0, 0, 255), -1)
            R_x_mean_meter, R_y_mean_meter=pixel_to_meter (R_win_x_mean, R_win_y_mean, origin_x, origin_y, S_x, S_y) # 좌표계 변환하고 저장
            R_points_meter_list.append((R_x_mean_meter+1.341, R_y_mean_meter))
            
            p_list.append((R_x_mean_meter+1.341, R_y_mean_meter+0.425)) # 오른쪽 차선기반 초록 데이터 포인트
            
            R_poly_func = R_Polyft_Plotting(R_points_meter_list, R_scatter, ax) # 오른쪽 차선 피팅용 함수 불러오기       
            R_normal_vector = R_normal_vector_cal(R_x_mean_meter+1.341, R_poly_func) 
            nv_p_list.append(   (R_x_mean_meter+1.341, R_y_mean_meter) + (R_normal_vector))  # 검정 차선. 튜플끼리 더하기

        else: break # 윈도우에서 데이터가 탐지 안 된다면 해당 프레임에서 윈도우 반복문 종료
    R_win_x_prev = R_win_x_current # 다음 프레임으로 넘어가기 전에 현재 값들을 이전 값으로 저장
    
    # 다음 프레임으로 넘어가기 전에 피팅
    if len(R_points_meter_list) > 0: # 오른쪽 차선 데이터포인트로 오른쪽 차선 피팅
        R_Polyft_Plotting(R_points_meter_list, R_scatter, ax)


    
    
    
    # 왼쪽 윈도우
    for window in range(L_num_windows):              
        if window == 0:
            if L_win_x_current > 20:
                L_win_x_current = 20
        L_window_height, L_win_y_low, L_win_y_high, L_win_x_low, L_win_x_high, L_win_indices, L_win_x_mean, L_win_y_mean = window_info(preprocessed_frame, L_num_windows, window, L_win_x_current, L_margin)
        if len(L_win_indices) > min_points:

            if preprocessed_frame[L_win_y_mean, L_win_x_mean] == 255:
                L_win_x_current = L_win_x_mean if abs(L_win_x_mean - L_win_x_prev) < L_consistency_threshold else L_win_x_prev
                cv2.rectangle(result_frame, (L_win_x_low, L_win_y_low), (L_win_x_high, L_win_y_high), (255, 0, 0), 2) # 파란색 윈도우랑 데이터 포인트
                cv2.circle(result_frame, (L_win_x_mean, L_win_y_mean), 7, (255, 0, 0), -1)
                L_x_mean_meter, L_y_mean_meter=pixel_to_meter (L_win_x_mean, L_win_y_mean, origin_x, origin_y, S_x, S_y) # 좌표계 변환하고 저장
                L_points_meter_list.append((L_x_mean_meter+1.341, L_y_mean_meter))   
            else:
                little_L_indices = np.column_stack(np.where(preprocessed_frame[L_win_y_low:L_win_y_high, :L_win_x_mean] > 0))
                if len(little_L_indices) > min_points:
                    little_L_win_x_mean = int(np.mean([L_win_x_low + idx2[1] for idx2 in little_L_indices]))
                    little_L_win_y_mean = int(np.mean([L_win_y_low + idx2[0] for idx2 in little_L_indices]))
                    # 초록색 윈도우랑 데이터 포인트
                    cv2.rectangle(result_frame, (L_win_x_low, L_win_y_low), (L_win_x_high, L_win_y_high), (0, 255, 0), 2)
                    cv2.circle(result_frame, (little_L_win_x_mean, little_L_win_y_mean), 7, (0, 255, 0), -1)
                    # 좌표계 변환하고 저장
                    L_little_x_mean_meter, L_little_y_mean_meter=pixel_to_meter (little_L_win_x_mean, little_L_win_y_mean, origin_x, origin_y, S_x, S_y)
                    L_points_meter_list.append((L_little_x_mean_meter+1.341, L_little_y_mean_meter))
                    


            L_poly_func = L_Polyft_Plotting(L_points_meter_list, L_scatter, ax)
            
            
            if len(R_points_meter_list) == 0:
                L_normal_vector = L_normal_vector_cal(L_x_mean_meter+1.341, L_poly_func)
                nv_p_list.append(   (L_x_mean_meter+1.341, L_y_mean_meter) + (L_normal_vector))  # 검정 차선. 튜플끼리 더하기                
        else: break # 윈도우에서 데이터가 탐지 안 된다면 해당 프레임에서 윈도우 반복문 종료
    L_win_x_prev = L_win_x_current # 다음 프레임으로 넘어가기 전에 현재 값들을 이전 값으로 저장
    
    
    
    # 다음 프레임으로 넘어가기 전에 피팅
    if len(L_points_meter_list) > 0: # 왼쪽 차선 피팅
        L_Polyft_Plotting(L_points_meter_list, L_scatter, ax) 
    
    
    if len(p_list) > 0: # 초록차선 피팅
        p_poly_func = p_Polyft_Plotting(p_list, p_scatter, ax)
        
    if len(nv_p_list) > 0: # 오른쪽 차선 데이터가 없다면(다시말해, 왼쪽 차선 데이터만 있다면) 1차 피팅, 있다면 3차 피팅
        if len(R_points_meter_list) == 0: N =1
        else: N = 3
        nv_p_poly_func = nv_p_Polyft_Plotting(nv_p_list, nv_p_scatter, ax, N)

    lookahead_point_list = lookahead_point_cal (0, 0, Ld, nv_p_poly_func, nv_p_list)
    if lookahead_point_list is not None:
        for lookahead_point in lookahead_point_list:
            
            ############# 디버깅##############
            for line in ax.lines:
                if line.get_label() == "lookahead_point":
                    line.remove()
            ############# 디버깅############## 
            
            ax.plot(lookahead_point[0], lookahead_point[1], 'o', color='orange', markeredgecolor='black', label="lookahead_point", markersize=10)
            
            # 해당 포인트에 대한 룩어헤드 벡터, 알파, 델타 계산
            lookahead_point_vec = lookahead_point_vec_cal(lookahead_point[0], lookahead_point[1])
            alpha_radians = angle_between_vectors(car_direcrtion_vec, lookahead_point_vec)
            delta_radians = delta_raidians_cal (alpha_radians)
            delta_degrees = math.degrees(delta_radians)
            
            
            print(f"alpha radians: {alpha_radians} \t delta radians: {delta_radians}  \t delta degrees: {delta_degrees}")
    
        

    
    canvas = FigureCanvas(fig) # 맷플롯립에서 생성된 fig(플롯 객체)를 캔버스에 연결
    canvas.draw() # 플롯을 캔버스에 랜더링
    
    
    plot_image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8') # 캔버스의 플롯 이미지를 RGBA형식으로 읽어오고, 넘파이 배열로 변환
    plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (4,)) # 캔버스의 너비와 높이를 받아와서, [::-1]로 순서 뒤집고, 마지막 채널에 4추가 해서 (높이, 너비, 채널) 형태로 만듦
    plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGBA2BGR) # opencv에서 호환 가능하게 BGR로 변경

    # OpenCV 창 출력
    cv2.imshow("Original Video", frame)
    cv2.imshow("Sliding Window Result", result_frame)
    cv2.imshow("Polynomial Fit Plot", plot_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
