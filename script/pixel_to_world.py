def pixel_to_meter(x_pixel, y_pixel, origin_x, origin_y, S_x, S_y):
    x_meter = (x_pixel - origin_x) * S_x
    y_meter = (origin_y - y_pixel) * S_y
    return y_meter, -x_meter # return x_meter, y_meter  결과를 x축대칭, y축대칭, y=x축 대칭을 한 상황