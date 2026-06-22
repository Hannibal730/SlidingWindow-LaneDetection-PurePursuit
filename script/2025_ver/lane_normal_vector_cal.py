import numpy as np

#########################################################################################
# 차선 평행이동 시킨 점들로 리스트 만들기 위해
# 접선벡터에 수직인 단위벡터를 구하고, 그 값에 평행이동시킬 거리(0.425)를 곱한 결과를 오른쪽 차선에 더해야지
#########################################################################################

def R_normal_vector_cal(point, R_poly_func): # 오른쪽 차선함수의 접선벡터에 수직인 단위벡터
    R_poly_prime = np.polyder(R_poly_func)  # 도함수
    
    R_tangent_slope = R_poly_prime(point) # point에서 접선 벡터의 미분값 구하기
    R_tangent_vector = np.array([1, R_tangent_slope]) # 접선 벡터
    
    R_normal_vector = np.array([-R_tangent_slope, 1]) # 접선 벡터에 수직인 법선 벡터
    R_normal_vector = R_normal_vector / np.linalg.norm(R_normal_vector)  # 단위 벡터로 변환
    R_normal_vector *= 0.425 
    
    return R_normal_vector


def L_normal_vector_cal(point, L_poly_func): # 왼쪽 차선함수의 접선벡터에 수직인 단위벡터
    L_poly_prime = np.polyder(L_poly_func)  # 도함수
    
    L_tangent_slope = L_poly_prime(point) # point에서 접선 벡터의 미분값 구하기
    L_tangent_vector = np.array([1, L_tangent_slope]) # 접선 벡터
    
    L_normal_vector = np.array([L_tangent_slope, -1]) # 접선 벡터에 수직인 법선 벡터
    L_normal_vector = L_normal_vector / np.linalg.norm(L_normal_vector)  # 단위 벡터로 변환
    L_normal_vector *= 0.425 
    
    return L_normal_vector