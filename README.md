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
4. **좌표계 변환**: 현실 세계에서의 제어를 위해서 픽셀 좌표계를 미터 좌표계로 변환
5. **슬라이딩 윈도우**: 개별 윈도우마다 흰 픽셀의 평균 좌표를 차선 포인트로 계산. (평균 좌표가 검정 픽셀 위인 경우는 별도 처리)
6. **다항식 피팅**: 차선 포인트를 `np.polyfit`로 다항식(1 or 3차) 모델에 피팅하여 좌/우 차선함수 생성
7. **법선벡터 평행이동**: 오른쪽 차선함수를 법선벡터 방향으로 평행이동하여 경로함수 생성
8. **룩어헤드 포인트**: 후륜축 중심 원과 경로함수의 교점을 `scipy.optimize.fsolve`로 계산. (교점 없다면 원의 반지름 점차 증가)
9.  **퓨어 퍼슛**: 벡터 내적 및 삼각함수를 이용해 조향각 계산
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




---

## 모듈 설명


### [image_preprocessing.py](image_preprocessing.py)  
- **`Bev_Gray_Blurred_Binary_transform(frame)`**  
  1. **Bird’s‑Eye View 변환**  
     - 원본 영상에서 도로 평면이 수직으로 보이도록 4개 기준점(src_pts → dst_pts)으로 BEV 매트릭스 생성
     - BEV 매트릭스와 `cv2.warpPerspective`로 BEV 영상을 생성

      | BEV 이전 | BEV 이후 |
      |:--------:|:--------:|
      | <img src="https://github.com/user-attachments/assets/6195e83f-2a52-4f74-a813-da455d5c44bd" width="600px" /> | <img src="https://github.com/user-attachments/assets/c7094304-f7a8-411a-bd52-c98b3879e42c" width="600px" /> |



  2. **그레이스케일 변환**  
     - 컬러(BGR) 프레임을 `cv2.cvtColor(BEV영상, COLOR_BGR2GRAY)`로 그레이스케일 변환  
  1. **가우시안 블러**  
     - `GaussianBlur` 커널 크기 (19×19), σ=0 으로 노이즈 제거

  1. **이진화 (Thresholding)**  
     - 픽셀 값이 225 이상이면 흰색(255), 그 외 검은색(0)으로 변환  
     - 결과는 검은색 바탕에 흰색 차선만 나타나는 영상


      | 가우시안 블러 없음     |  가우시안 블러 처리   |
      |:--------:|:--------:|
      | ![Image](https://github.com/user-attachments/assets/f7fee490-c1a5-426d-91f8-d7b37c7c4b23)     | ![Image](https://github.com/user-attachments/assets/f519e5f9-422b-467b-bbce-a77733ef850c)    | 





---






### `lane_histogram.py`  
- **`histogram_argmax(frame)`**  
  1. **하단부 ROI 지정**  
     - 전체 높이의 아래 약 2/3 지점(1.15·h/3)부터 마지막 행까지 사용  
  2. **히스토그램 계산**  
     - ROI 내 모든 column별로 흰색 픽셀 합산하여 히스토그램 생성
       ![Image](https://github.com/user-attachments/assets/63d69952-a237-472a-946e-b6d05d713e44) 
  3. **좌/우 차선 시작점 계산**  
     - 하스토그램을 절반으로 나눠 좌/우 부분에서 각각 `np.argmax` 계산하고, 각각 L_x, R_x 로 반환
     - L_x, R_x는 최초 바운딩 박스 시작지점으로 사용  
  4. **유효 범위 threshold 설정**  
     - 히스토그램 길이·1/3, 1.6/3 지점에서 L_x와 R_x의 임계값(L_thr, R_thr) 계산
     




---






### `pixel_to_world.py`  
- **`pixel_to_meter(x_px, y_px, origin_x, origin_y, S_x, S_y)`**  
  - **원점(origin_x, origin_y)**: 영상 내 차량 위치 기준 픽셀 좌표  
  - **스케일링(S_x, S_y)**: 픽셀 단위 거리 → 실측(m) 비율  
  - 공식:  
    ```python
    x_meter = (x_px - origin_x) * S_x  
    y_meter = (origin_y - y_px) * S_y 
    return y_meter, -x_meter
    ```  


    | 픽셀 단위                     | 미터 단위                                                          |
    |:--------:|:--------:|
    | ![Image](https://github.com/user-attachments/assets/732b095c-e626-4de1-b454-6d935d45b683)| ![Image](https://github.com/user-attachments/assets/a19ab31c-32f9-45ca-a16d-3cc2766cb058) |   





---

### `sliding_window.py`  
- **`SlidingWindow` 클래스**  
  - 이진화된 프레임에서 하단 히스토그램을 구해 초기 좌/우 차선 위치 추정  
  - 슬라이딩 윈도우 반복으로 차선 픽셀 수집 → 픽셀→미터 좌표 변환 → 좌/우 차선의 포인트 리스트 (p_pts) 생성
  - 슬라이딩 윈도우로 생성한 차선 포인트가 검정 픽셀일 경우, 그 픽셀을 기준으로 더 외곽에서 흰색 픽셀 중 선택

    | 검정픽셀 선택     |  흰섹 픽셀 선택   |
    |:--------:|:--------:|
    | ![Image](https://github.com/user-attachments/assets/b682be67-b525-4ced-b765-cbc6237eac26)     | ![Image](https://github.com/user-attachments/assets/825be0c3-d01c-4f89-816e-0b945a4ffa68)   | 





  - p_pts를 polynomial fitting 처리하여 좌/우 차선함수 생성. (홀수 차수 사용)
- 평행이동
  - 왼쪽 차선은 점선이라서 포인트 개수가 부족하다. 반면에 오른쪽 차선은 실선이라서 포인트 개수가 풍족하다. 따라서 오른쪽 차선을 평행이동하여 경로함수를 생성한다.
  
- 평행이동 시 법선벡터의 필요성
  - 차선함수를 단순 평행이동하여 경로함수를 제작할 경우에 왜곡이 발생한다. 따라서 차선함수의 법선벡터를 계산하고, 차선함수를 법선벡터 방향으로 평행이동하여 경로함수를 제작해야 한다. 
  
    ![Image](https://github.com/user-attachments/assets/9a3602e4-025c-497c-8359-2f5c6f5ea345)
  
  
  - p_pts를 법선벡터 방향으로 평행이동하여 nv_pts(normal vextor points list) 를 제작하고, polinormial fitting 처리하여 경로함수를 제작한다.
   
- **`process(binary_frame, ax, L_sc, R_sc, p_sc, nv_sc)`**  
  1. `histogram_argmax` 호출 → `Lx_arg`, `Rx_arg`, 임계치 계산  
  2. `window_info`로 각 윈도우의 픽셀 좌표 및 평균 좌표 획득  
  3. `pixel_to_meter`로 월드 좌표 변환 → `R_pts_m`, `L_pts_m`, `p_pts`, `nv_pts` 리스트 적재  
  4. `R_Polyft_Plotting`·`L_Polyft_Plotting`·`R_normal_vector_cal` 호출  
  5. 최종 `result_frame`(컬러 윈도우 표시)와 4개 포인트 리스트를 반환  


---


### `polynomial_fit.py`  
- **`R_Polyft_Plotting`, `L_Polyft_Plotting`**  
  - 오른쪽 차선은 포인트 개수가 충분하기 때문에 3차 다항식으로, 왼쪽 차선은 포인트 개수가 부족하기 때문에 1차 다항식으로 `np.polyfit` 진행
  - `np.poly1d` 반환하며, 플롯 위에 피팅 곡선(300개 점)을 실시간 업데이트  
- **`p_Polyft_Plotting`, `nv_p_Polyft_Plotting`**  
  - 단순 평행이동 버전 중간 경로함수(Path)와 법선 벡터 보정한 버전의 경로함수(nv Path)에 대해 3차 다함식 피팅  
- 

---


### `lane_normal_vector_cal.py`  
- **`R_normal_vector_cal(x, R_poly_func)`** / **`L_normal_vector_cal(x, L_poly_func)`**  
  1. 주어진 X 좌표에서 다항식의 도함수를 `np.polyder`로 구함  
  2. 접선 기울기(m) → 벡터 `[1, m]` 생성  
  3. 법선 벡터 `[-m, 1]` (또는 `[m, -1]`)을 단위 벡터로 정규화  
  4. 단위 벡터에 도로 폭의 절반인 ** 0.425 m**를 법선벡터에 곱해, 경로함수를 위한 법선벡터를 제작한다.



---

### `pure_pursuit.py`  
- **`lookahead_point_cal(a, b, r, poly_func, nv_pts)`**  
  1. 차량 중심 (a,b)에서 반지름 r 원 방정식과 `poly_func(x)` 교점 찾기(`fsolve`)  
  2. 초기 r = `Ld(=2.5)`m 시도 → 실패 시 최대 10m까지 1m 단위로 반지름 증가 재시도  
  3. 양수 X 해만 남겨 `[(x_sol, y_sol)]` 형태 리스트 반환  
- **`lookahead_point_vec_cal(pt_x, pt_y)`**  
  - 룩어헤드 포인트로부터 차량 후륜축 중심까지의 벡터 계산  (거리는 1.341m 고정)
- **`angle_between_vectors(v1, v2)`**  
  - 두 벡터 내적/크기 → 코사인값 → `arccos` → 부호(우회전 시 음수) 적용  
- **`delta_raidians_cal(alpha)`**  
  - Pure Pursuit δ = arctan(2·L·sin α / Ld) 공식으로 최종 조향각 계산 

---

### `visualization.py`  
- **`create_lane_plot()`**  
  - 파란 점, 실선: 왼쪽 차선 포인트, 1차 다항식으로 피팅된 왼쪽 차선 함수
  - 빨간 점, 실선: 오른쪽 차선 포인트, 3차 다항식으로 피팅된 오른쪽 차선 함수
  - 초록 점, 실선: 법선벡터 보정을 하지 않고 단순히 오른쪽 차선 함수를 평행이동한 포인트, 경로함수
  - 검정 점, 실선: 법센벡터 보정한 채로 오른쪽 차선 함수를 평행이동한 포인트, 경로함수
  - 노란 점: 룩어헤드 포인트
  - 보라 사각형: 차량 후륜축 중심
  

    |     ![Image](https://github.com/user-attachments/assets/4dabb9b8-aca6-42e0-9a72-182aedb0d15a) |     ![Image](https://github.com/user-attachments/assets/77bfb545-1fea-4349-996c-131a678b2f15)  |
    |:--------:|:--------:|








    ---

### `main.py`  
- **스크립트 흐름**  
  1. **모듈 임포트 & 초기 설정**  
     - OpenCV 비디오 캡처, Matplotlib 플롯 생성  
     - `sliding_window.SlidingWindow`, `pure_pursuit.PurePursuit` 인스턴스화  
  2. **프레임 루프**  
     ```python
     while cap.isOpened():
         ret, frame = cap.read()
         if not ret: break
         binary = Bev_Gray_Blurred_Binary_transform(frame)
         # 슬라이딩 윈도우 탐색
         result, R_pts, L_pts, p_pts, nv_pts = sw_detector.process(
             binary, ax, L_scatter, R_scatter, p_scatter, nv_p_scatter
         )
         # 경로·법선 다항식 피팅
         if p_pts:    p_Polyft_Plotting(p_pts, p_scatter, ax)
         if nv_pts:
             degree = 1 if not R_pts else 3
             nv_poly = nv_p_Polyft_Plotting(nv_pts, nv_p_scatter, ax, degree)
         # Pure Pursuit 룩어헤드 & 조향각 계산
         if nv_poly:
             lookahead_pts = lookahead_point_cal(0, 0, Ld, nv_poly, nv_pts)
             # 시각화 및 angle 계산
         # Matplotlib → OpenCV 이미지 변환 후 화면에 출력
     ```
  3. **종료 처리**  
     - `cap.release()`, `cv2.destroyAllWindows()` 호출  

---

## 멘토
건국대학교 로봇동아리 돌밭 자율주행팀 임현우


