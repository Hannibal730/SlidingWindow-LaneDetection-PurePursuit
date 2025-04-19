# plot_setup.py

import matplotlib.pyplot as plt

def create_lane_plot():
    """
    Lane Points와 Polynomial Fitting을 그리기 위한 Matplotlib Figure와 Axes,
    그리고 빈 scatter Line2D 객체들을 초기화하여 반환합니다.
    """
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=False)

    # 그리드 및 축 레이블 설정
    ax.grid(True)
    ax.set_xlim(0, 2.0 + 1.341)    # X축 범위 (0m ~ 2m + offset)
    ax.set_ylim(-1.2, 1.2)        # Y축 범위 (-0.7m ~ 0.7m)
    ax.set_title("Lane Points and Polynomial Fitting in Meter Coordinates")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")

    # 차량 후륜축 중심 위치 표시용 네모
    rear_axle_center_size = 0.07
    rear_axle_center = plt.Rectangle(
        (0, -0.5 * rear_axle_center_size),
        rear_axle_center_size,
        rear_axle_center_size,
        linewidth=1, edgecolor='purple', facecolor='purple'
    )
    ax.add_patch(rear_axle_center)
    ax.plot([], [], 's', color="purple", label="rear_axle_center")

    # 빈 scatter/line 객체들
    L_scatter, = ax.plot([], [], 'bo', label="Left Lane Points")
    ax.plot([], [], '-', color="blue", label="Left Lane Fitting")

    R_scatter, = ax.plot([], [], 'ro', label="Right Lane Points")
    ax.plot([], [], '-', color="red", label="Right Lane Fitting")

    p_scatter, = ax.plot([], [], 'o', color='green', label="Path Points")
    ax.plot([], [], '-', color="green", label="Path Fitting")

    nv_p_scatter, = ax.plot([], [], 'o', color='black', label="nv Path Points")
    ax.plot([], [], '-', color="black", label="nv Path Fitting")

    lookahead_scatter, = ax.plot([], [], 'o', color='orange', label="Lookahead Point")

    # 범례 및 레이아웃
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1), borderaxespad=0.)
    plt.subplots_adjust(right=0.9)
    plt.tight_layout()

    # 반환: fig, ax, 그리고 scatter 핸들들
    return {
        'fig': fig,
        'ax': ax,
        'L_scatter': L_scatter,
        'R_scatter': R_scatter,
        'p_scatter': p_scatter,
        'nv_p_scatter': nv_p_scatter,
        'lookahead_scatter': lookahead_scatter,
    }
