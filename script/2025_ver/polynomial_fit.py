import numpy as np


#########################################################################################
# 피팅 함수 설정
#########################################################################################
def R_Polyft_Plotting(R_points_meter_list, R_scatter, ax):
    R_x, R_y = zip(*R_points_meter_list)
    R_scatter.set_data(R_x, R_y)
    # 일반 다항식 피팅 (3차 다항식)
    R_coeff = np.polyfit(R_x, R_y, 3)
    R_poly_func = np.poly1d(R_coeff)
    # x 범위에 대한 피팅된 y 값 계산
    R_x_fit = np.linspace(min(R_x), max(R_x), 300)
    R_y_fit = R_poly_func(R_x_fit)
    
    
    # 이전 피팅 선 제거
    for line in ax.lines:
        if line.get_label() == "Right Lane Fit":
            line.remove()
    # 새로운 피팅 선 추가
    ax.plot(R_x_fit, R_y_fit, color="red", label="Right Lane Fit")
    return R_poly_func


def L_Polyft_Plotting(L_points_meter_list, L_scatter, ax):
    L_x, L_y = zip(*L_points_meter_list)
    L_scatter.set_data(L_x, L_y)
    # 일반 다항식 피팅 (1차 다항식)
    L_coeff = np.polyfit(L_x, L_y, 1)
    L_poly_func = np.poly1d(L_coeff)
    # x 범위에 대한 피팅된 y 값 계산
    L_x_fit = np.linspace(min(L_x), max(L_x), 300)
    L_y_fit = L_poly_func(L_x_fit)
    

    
    # 이전 피팅 선 제거
    for line in ax.lines:
        if line.get_label() == "Leftt Lane Fit":
            line.remove()
    # 새로운 피팅 선 추가
    ax.plot(L_x_fit, L_y_fit, color="blue", label="Leftt Lane Fit")
    return L_poly_func




def p_Polyft_Plotting(p_list, p_scatter, ax):
    if len(p_list) ==0:
        return None
    p_x, p_y = zip(*p_list)
    p_scatter.set_data(p_x, p_y)
    # 일반 다항식 피팅 (3차 다항식)
    p_coeff = np.polyfit(p_x, p_y, 3)
    p_poly_func = np.poly1d(p_coeff)
    # x 범위에 대한 피팅된 y 값 계산
    p_x_fit = np.linspace(min(p_x), max(p_x), 300)
    p_y_fit = p_poly_func(p_x_fit)
    # 이전 피팅 선 제거
    for line in ax.lines:
        if line.get_label() == "Path Fit":
            line.remove()
    # 새로운 피팅 선 추가
    ax.plot(p_x_fit, p_y_fit, color="green", label="Path Fit")
    return p_poly_func


def nv_p_Polyft_Plotting(nv_p_list, nv_p_scatter, ax, N):
    if len(nv_p_list) == 0:
        return None
    global nv_p_x
    nv_p_x, nv_p_y = zip(*nv_p_list)
    nv_p_scatter.set_data(nv_p_x, nv_p_y)
    # 일반 다항식 피팅 (3차 다항식)
    nv_p_coeff = np.polyfit(nv_p_x, nv_p_y, N)
    nv_p_poly_func = np.poly1d(nv_p_coeff)
    
    nv_p_x_fit = np.linspace(min(nv_p_x), max(nv_p_x), 300) # 임의의 x범위 설정
    nv_p_y_fit = nv_p_poly_func(nv_p_x_fit) # x 범위에 대한 피팅된 y 값 계산
    

    
    # 이전 피팅 선 제거
    for line in ax.lines:
        if line.get_label() == "nv Path Fit":
            line.remove()
    # 새로운 피팅 선 추가
    ax.plot(nv_p_x_fit, nv_p_y_fit, color="black", label="nv Path Fit")
    return nv_p_poly_func
