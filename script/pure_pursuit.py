import numpy as np
import math
from scipy.optimize import fsolve


# #########################################################################################
# # 중심과 반지름이 (a,b), r인 원과 차선경로 피팅함수와의 교점(룩어헤드 포인트)을 계산하는 함수
# #########################################################################################
Ld = 2.5
def lookahead_point_cal (a, b, r, nv_p_poly_func, nv_p_list):
    def circle_equation (x, r):
        return (x-a)**2 + (nv_p_poly_func(x)-b)**2 -r**2
    
    # 초기화
    if len (nv_p_list) == 0: return []
    x_nv_p_list = list(zip(*nv_p_list))[0]
    lookahead_point_list = []
    
    
    x_sol = fsolve(circle_equation, x_nv_p_list, args=(Ld,)) # 반지름 길이가 Ld일 때, 교점의 x좌표인 x_sol 찾기

    
    if len(x_sol) > 0 and not np.isnan(x_sol).any(): # x_sol이 존재하면
        print("Lookahead Point can be calculated")
        x_sol = [x for x in x_sol if x > 0]
        y_sol = nv_p_poly_func(x_sol)
        lookahead_point_list.append((x_sol, y_sol))
        

    else: # x_sol이 존재 안 하면
    # 반지름 길이 r  반복문
        print("Lookahead Point can not be calculated")
        max_r = 10.0   # 최대 탐색 반지름
        for i in np.arange(r, max_r, 1.0):  # r → max_r까지 1씩 증가
            x_sol = fsolve(circle_equation, x_nv_p_list, args=(i,))
            if len(x_sol) > 0 and not np.isnan(x_sol).any():
                y_sol = nv_p_poly_func(x_sol)
                lookahead_point_list.append((x_sol, y_sol))
                break

    return lookahead_point_list







#########################################################################################
# 차량의 정면방향과 룩어헤드 포인트 방향 사이의 사이각 계산
#########################################################################################

car_direcrtion_vec = [1, 0]
def lookahead_point_vec_cal (a, b): # 룩어헤드 포인트 좌표 (a, b)
    # lookahead_point_vec = [a-1.341, b-0]
    lookahead_point_vec = [a[0]-1.341, b[0]-0]
    return lookahead_point_vec


def angle_between_vectors(vec1, vec2): # 입력할 순서는 차량 직선방향, 룩어헤드 포인트
    dot_product = np.dot(vec1, vec2) # 내적
    magnitude1 = np.linalg.norm(vec1) # 두 벡터의 크기 계산
    magnitude2 = np.linalg.norm(vec2)
    
    cos_theta = dot_product / (magnitude1 * magnitude2) # 두 벡터 사이의 코사인 값 계산
    angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0)) # 코사인 값을 이용해 각도 계산 (라디안 단위) # np.clip이 코사인 값을 -1 ~ 1 범위로 제한
    
    if vec2[1] <0: # 룩어헤드 포인트 벡터의 y좌표가 차량직선방향의 y좌표인 0보다 작다면 우회전 필요상황
        angle_radians *= -1 # 우회전 필요상황에선 조향각의 라디안 값이 음수로 나오게
    
    return angle_radians


#########################################################################################
# 조향각 계산
#########################################################################################
L = 0.55 # 휠 베이스
def delta_raidians_cal (alpha_radians):
    delta_radians = math.atan (2*L*math.sin(alpha_radians) / Ld)
    return delta_radians




