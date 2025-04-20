# Lane Detection & Pure Pursuit for Steering Angle of Local Path Planning

로컬 주행 영상을 입력받아 **슬라이딩 윈도우** 기반 차선 검출 기법과 **퓨어 퍼슈트(Pure Pursuit)** 알고리즘을 결합하는 프로젝트입니다. 이를 통해 실시간으로 조향각을 산출하고, 경로함수를 생성·시각화합니다.



1. 실행 시작
   ![Image](https://github.com/user-attachments/assets/1dee7d03-a1b1-4b49-aaca-eaaea939b834)

2. 좌회전
    ![Image](https://github.com/user-attachments/assets/8636e5e5-9f0b-404a-b9ab-b0bb32dffca7)

3. 횡단보도
    ![Image](https://github.com/user-attachments/assets/c159765d-024a-43fe-8005-710e6d107a34)

4. 우회전
    ![Image](https://github.com/user-attachments/assets/4c0ecdfd-d7ca-42ff-97c1-b83d36f300b8)



## 작동 순서

이 프로젝트는 다음 단계를 순차적으로 수행합니다:

1. **영상 입력**: `cv2.VideoCapture`로 지정 경로의 비디오를 읽고, 프레임 단위로 처리
2. **전처리**: Bird’s Eye View(투시 변환) → 그레이스케일 → 가우시안 블러 → 이진화
3. **초기 차선 위치 추정**: 영상 하단부 흰 픽셀 히스토그램을 이용해 좌/우 차선의 대략적 x좌표 계산
4. **슬라이딩 윈도우**: 각 윈도우에서 흰 픽셀을 추적, 평균 좌표 산출 → 픽셀 좌표계에서 미터 좌표계로 변환
5. **다항식 피팅**: `np.polyfit`로 차선 및 주행 경로에 다항식(1~3차) 모델 적합
6. **룩어헤드**: 차량 전방 원과 경로 다항식 교점을 `scipy.optimize.fsolve`로 풀이
7. **퓨어 퍼슈트**: 벡터 내적 및 삼각함수를 이용해 조향각 계산
8. **시각화**: OpenCV (`imshow`) + Matplotlib(`FigureCanvas`)으로 결과 출력

이를 통해 차선 기반 주행 경로를 실시간으로 생성하고, 해당 경로를 따라 차량 조향 각도를 산출합니다.

---

## 환경설치 및 실행방법

```bash
# Python 3.7 이상
pip install opencv-python numpy matplotlib scipy
```

```bash
python main.py
```

---




## 주요 파라미터

| 변수                     | 설명                                                          | 기본값       |
|-------------------------|--------------------------------------------------------------|------------|
| `origin_x, origin_y`    | 픽셀 좌표계에서 미터 좌표계의 원점. 예: `(569-28)/2, 480`     | `(270.5,480)`|
| `actuaL_lane_width`     | 실제 차선 간 폭 (m)                                           | `0.85`     |
| `lane_pixel_width`      | 영상 상 차선 폭 픽셀 길이 (`569-28`)                         | `541`      |
| `actuaL_lane_length`    | 차선 길이 기준 (m)                                           | `0.5`      |
| `lane_pixel_length`     | 픽셀 길이 (`247-123`)                                        | `124`      |
| `S_x, S_y`              | 픽셀→미터 변환 비율: `actuaL_lane_width/lane_pixel_width`, `actuaL_lane_length/lane_pixel_length` | 계산 자동 |
| `R_num_windows, L_num_windows` | 윈도우 개수 (오른쪽 25, 왼쪽 6)                 | `25, 6`    |
| `R_margin, L_margin`    | 윈도우 반폭 (픽셀)                                            | `60, 150`  |
| `min_points`            | 윈도우 내 최소 검사 픽셀 수                                    | `10`       |
| `R_consistency_threshold, L_consistency_threshold` | x좌표 이동 한계 (픽셀)           | `560,600`  |
| `Ld`                    | 룩어헤드 탐색 반경 (m)                                        | `2.5`      |
| `L`                     | 휠베이스 (m)                                                 | `0.55`     |

---

## 모듈 설명


### `image_preprocessing.py`  
- **`Bev_Gray_Blurred_Binary_transform(frame)`**  
  Bird’s‑Eye View 투시 변환 → 그레이스케일 → 가우시안 블러 → 이진화 처리  
  → 0/255 바이너리 프레임 반환 :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}  

### `lane_histogram.py`  
- **`histogram_argmax(frame)`**  
  영상 하단 2/3 영역의 흰 픽셀 합산 히스토그램 생성 →  
  좌/우 최대값 X 인덱스 및 유효 임계치 반환 :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}  

### `pixel_to_world.py`  
- **`pixel_to_meter(x_px, y_px, origin_x, origin_y, S_x, S_y)`**  
  픽셀 좌표 → 실제 거리(m) 변환 (원점, 스케일링 파라미터 사용) :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}  

### `lane_normal_vector_cal.py`  
- **`R_normal_vector_cal(x, R_poly)`**, **`L_normal_vector_cal(x, L_poly)`**  
  차선 다항식에서 접선의 법선 단위벡터 계산 → 0.425 m 만큼 평행 이동 벡터 반환 :contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7}  

### `polynomial_fit.py`  
- **`R_Polyft_Plotting`**, **`L_Polyft_Plotting`**, **`p_Polyft_Plotting`**, **`nv_p_Polyft_Plotting`**  
  1차~3차 다항식으로 차선 및 경로 곡선 피팅 →  
  Matplotlib 플롯 업데이트 후 `np.poly1d` 함수 반환 :contentReference[oaicite:8]{index=8}&#8203;:contentReference[oaicite:9]{index=9}  

### `pure_pursuit.py`  
- **`lookahead_point_cal(a, b, r, poly_func, nv_pts)`**  
  차량 중심 (a,b), 반지름 r 원과 경로 곡선의 교점(룩어헤드 포인트) 계산  
- **`lookahead_point_vec_cal`**, **`angle_between_vectors`**, **`delta_raidians_cal`**  
  벡터 변환 → 각도 계산 → Pure Pursuit 공식에 따른 조향각(라디안) 산출 :contentReference[oaicite:10]{index=10}&#8203;:contentReference[oaicite:11]{index=11}  

### `visualization.py`  
- **`create_lane_plot()`**  
  Matplotlib Figure/Axis 생성, 축 범위 및 레이블 설정, 차량 후륜 축 표시,  
  좌/우 차선·경로·룩어헤드 포인트를 위한 빈 Scatter/Line2D 객체 초기화 후 반환 :contentReference[oaicite:12]{index=12}&#8203;:contentReference[oaicite:13]{index=13}  

### `main.py`  
- **`main()`**  
  1. 비디오 캡처  
  2. 프레임별: 전처리 → 히스토그램 → 윈도우 처리 → 좌표 변환 → 피팅 → Pure Pursuit → 시각화  
  3. 종료 시 리소스 해제  
