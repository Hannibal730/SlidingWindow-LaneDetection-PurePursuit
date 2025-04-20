# Lane Detection & Pure Pursuit for Steering Angle of Local Path Planning

로컬 주행 영상을 입력받아 **슬라이딩 윈도우** 기반 차선 검출 기법과 **퓨어 퍼슈트(Pure Pursuit)** 알고리즘을 결합하는 프로젝트입니다. 이를 통해 차선 기반 주행 경로를 실시간으로 생성하고, 해당 경로를 따라 차량 조향 각도를 산출합니다.




### 1. 직선구간
![Image](https://github.com/user-attachments/assets/1dee7d03-a1b1-4b49-aaca-eaaea939b834)

<br>

### 2. 좌회전
![Image](https://github.com/user-attachments/assets/8636e5e5-9f0b-404a-b9ab-b0bb32dffca7)

<br>

### 3. 횡단보도
![Image](https://github.com/user-attachments/assets/c159765d-024a-43fe-8005-710e6d107a34)

<br>

### 4. 우회전
![Image](https://github.com/user-attachments/assets/4c0ecdfd-d7ca-42ff-97c1-b83d36f300b8)

---

## 작동 과정

이 프로젝트는 다음 단계를 순차적으로 수행합니다

1. **영상 입력**: `cv2.VideoCapture`로 로컬 경로의 비디오(640x480)를 읽고, 프레임 단위로 처리
2. **전처리**: Bird’s Eye View 변환 → 그레이스케일 → 가우시안 블러 → 이진화
3. **초기 차선 위치 추정**: 히스토그램으로 영상 하단부 흰 픽셀 (차선)의 분포를 계산 → 좌/우 차선의 시작지점 계산
4. **슬라이딩 윈도우**: 개별 윈도우마다 흰 픽셀의 평균 좌표를 차선 포인트로 계산. (평균 좌표가 검정 픽셀 위인 경우는 별도 처리)
5. **좌표계 변환**: 현실 세계에서의 제어를 위해서 픽셀 좌표계를 미터 좌표계로 변환
6. **다항식 피팅**: 차선 포인트를 `np.polyfit`로 다항식(1 or 3차) 모델에 피팅하여 좌/우 차선함수 생성
7. **법선벡터 평행이동**: 오른쪽 차선함수를 법선벡터 방향으로 평행이동하여 경로함수 생성
8. **룩어헤드 포인트**: 후륜축 중심 원과 경로함수의 교점을 `scipy.optimize.fsolve`로 계산. (교점 없다면 원의 반지름 점차 증가)
9. **퓨어 퍼슛**: 벡터 내적 및 삼각함수를 이용해 조향각 계산
10. **시각화**: 원본 영상, 슬라이딩 윈도우 영상, 차선 및 경로함수 영상, 조향각을 동시에 출력


---

<br>

## 실행방법


**1. Git clone**
```bash
git clone https://github.com/Hannibal730/SlidingWindow-LaneDetection-PurePursuit.git
```

**2. 환경설치**

```bash
# Python 3.7 이상
pip install opencv-python numpy matplotlib scipy
```

**3. main코드 실행**
```bash
python main.py
```



---

<br>


## 주요 파라미터

| 변수                     | 설명                                                          | 기본값       |
|-------------------------|--------------------------------------------------------------|------------|
| `origin_x, origin_y`    | 픽셀 좌표계에서 미터 좌표계의 원점. 예: `(569-28)/2, 480`     | `(270.5,480)`|
| `actuaL_lane_width`     | 실제 차선 간 폭 (m)                                           | `0.85`     |
| `lane_pixel_width`      | 영상 상 차선 폭 픽셀 길이 (`569-28`)                         | `541`      |
| `actuaL_lane_length`    | 차선 길이 기준 (m)                                           | `0.5`      |
| `lane_pixel_length`     | 픽셀 길이 (`247-123`)                                        | `124`      |
| `S_x, S_y`              | 픽셀→미터 변환 비율: `actuaL_lane_width/lane_pixel_width`, `actuaL_lane_length/lane_pixel_length` | 0.5m / (247-123), 0.85m  <br>  (569-28) |
| `R_num_windows, L_num_windows` | 윈도우 개수 (오른쪽 25, 왼쪽 6)                 | `25, 6`    |
| `R_margin, L_margin`    | 윈도우 반폭 (픽셀)                                            | `60, 150`  |
| `min_points`            | 윈도우 내 최소 검사 픽셀 수                                    | `10`       |
| `R_consistency_threshold, L_consistency_threshold` | x좌표 이동 한계 (픽셀)           | `560,600`  |
| `Ld`                    | 룩어헤드 탐색 반경 (m)                                        | `2.5`      |
| `L`                     | 휠베이스 (m)                                                 | `0.55`     |


![Image](https://github.com/user-attachments/assets/a19ab31c-32f9-45ca-a16d-3cc2766cb058)
![Image](https://github.com/user-attachments/assets/732b095c-e626-4de1-b454-6d935d45b683)



| 미터 단위                     | 픽셀 단위                                                          |
|-------------------------|--------------------------------------------------------------|
| ![Image](https://github.com/user-attachments/assets/a19ab31c-32f9-45ca-a16d-3cc2766cb058) | ![Image](https://github.com/user-attachments/assets/732b095c-e626-4de1-b454-6d935d45b683)   |

---

## 모듈 설명


### `image_preprocessing.py`  
- **`Bev_Gray_Blurred_Binary_transform(frame)`**  
  1. **투시 변환 (Bird’s‑Eye View)**  
     - 원본 영상에서 도로 평면이 수직으로 보이도록 4개 기준점(src_pts → dst_pts)으로 투시 매트릭스를 계산  
     - `cv2.warpPerspective`로 왜곡 보정된 BEV 영상을 생성  
  2. **그레이스케일 변환**  
     - 컬러(BGR) 프레임을 `cv2.cvtColor(..., COLOR_BGR2GRAY)`로 회색조 이미지로 변경  
  3. **가우시안 블러**  
     - `GaussianBlur` 커널 크기 (19×19), σ=0 으로 노이즈 제거 및 디테일 부드럽게 처리  
  4. **이진화 (Thresholding)**  
     - 픽셀 값이 225 이상이면 흰색(255), 그 외 검은색(0)으로 변환  
     - 결과는 차선 픽셀만 강조된 0/255 바이너리 프레임  

---

### `lane_histogram.py`  
- **`histogram_argmax(frame)`**  
  1. **하단부 ROI 지정**  
     - 전체 높이의 아래 약 2/3 지점(1.15·h/3)부터 마지막 행까지 사용  
  2. **히스토그램 계산**  
     - ROI 내 모든 열별 흰색 픽셀 합산 → 1차원 배열(hist)  
  3. **좌/우 피크 인덱스**  
     - 배열을 절반으로 나눠 좌/우 부분에서 각각 `np.argmax` 호출  
     - 반환값 L_x, R_x는 초기 차선 윈도우 중앙 X 좌표로 사용  
  4. **유효 범위 임계치 설정**  
     - 히스토그램 길이·1/3, 1.6/3 지점에서 좌/우 임계값(L_thr, R_thr) 계산  

---

### `pixel_to_world.py`  
- **`pixel_to_meter(x_px, y_px, origin_x, origin_y, S_x, S_y)`**  
  - **원점(origin_x, origin_y)**: 영상 내 차량 위치 기준 픽셀 좌표  
  - **스케일링(S_x, S_y)**: 픽셀 단위 거리 → 실측(m) 비율  
  - 공식:  
    ```python
    x_meter = (x_px - origin_x) * S_x  
    y_meter = (origin_y - y_px) * S_y  
    # 최종 리턴은 (forward, lateral)로 y,x축을 서로 반전
    return y_meter, -x_meter
    ```  

---

### `lane_normal_vector_cal.py`  
- **`R_normal_vector_cal(x, R_poly_func)`** / **`L_normal_vector_cal(x, L_poly_func)`**  
  1. 주어진 X 좌표에서 다항식의 도함수를 `np.polyder`로 구함  
  2. 접선 기울기(m) → 벡터 `[1, m]` 생성  
  3. 법선 벡터 `[-m, 1]` (또는 `[m, -1]`)을 단위 벡터로 정규화  
  4. 지정된 **평행 이동 거리 0.425 m**를 곱해, 차선 중심 대신 차량 경로 중간 위치 산출  

---

### `polynomial_fit.py`  
- **`R_Polyft_Plotting`, `L_Polyft_Plotting`**  
  - 각각 3차(오른쪽), 1차(왼쪽) 다항식 조건으로 `np.polyfit` 수행  
  - `np.poly1d` 반환하며, 플롯 위에 피팅 곡선(300개 점)을 실시간 업데이트  
- **`p_Polyft_Plotting`, `nv_p_Polyft_Plotting`**  
  - 중간 경로(Path)와 법선 벡터 경로(nv Path)에 대해 3차 또는 가변 차수(N) 피팅  
  - Matplotlib 선 제거/추가 로직 포함  

---

### `pure_pursuit.py`  
- **`lookahead_point_cal(a, b, r, poly_func, nv_pts)`**  
  1. 차량 중심 (a,b)에서 반지름 r 원 방정식과 `poly_func(x)` 교점 찾기(`fsolve`)  
  2. 초기 r = `Ld(=2.5)`m 시도 → 실패 시 최대 10m까지 1m 단위로 반지름 증가 재시도  
  3. 양수 X 해만 남겨 `[(x_sol, y_sol)]` 형태 리스트 반환  
- **`lookahead_point_vec_cal(pt_x, pt_y)`**  
  - 룩어헤드 좌표로부터 차량 기준점(오프셋 1.341m) 차이 벡터 계산  
- **`angle_between_vectors(v1, v2)`**  
  - 두 벡터 내적/크기 → 코사인값 → `arccos` → 부호(우회전 시 음수) 적용  
- **`delta_raidians_cal(alpha)`**  
  - Pure Pursuit δ = arctan(2·L·sin α / Ld) 공식으로 최종 조향각 도출  

---

### `visualization.py`  
- **`create_lane_plot()`**  
  1. `plt.subplots`로 Figure/Axis 생성, 크기 11×6  
  2. X(0→≈3.34m), Y(−1.2→1.2m) 축 범위 및 그리드 설정  
  3. 차량 후륜축 위치를 보라색 사각형으로 표시  
  4. 빈 Scatter/Line2D 객체(좌/우 차선, 경로, 법선 경로, 룩어헤드)를 반환  

---

### `main.py`  
- **`main()`**  
  1. `cv2.VideoCapture`로 비디오 파일 열기  
  2. 프레임별 반복  
     - 전처리 → 히스토그램 → 슬라이딩 윈도우로 픽셀 탐색  
     - 월드 좌표 변환 → 다항식 피팅 → 법선 벡터 경로 생성  
     - Pure Pursuit 룩어헤드 포인트 & 조향각 계산  
     - 원본·윈도우·피팅 플롯을 실시간 갱신  
  3. 종료 시 `cap.release()`, `cv2.destroyAllWindows()` 호출  




---
