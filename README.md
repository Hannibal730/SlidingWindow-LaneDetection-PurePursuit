# Lane Detection & Pure Pursuit for Steering Angle of Local Path Planning

로컬 주행 영상을 입력받아 **슬라이딩 윈도우** 기반 차선 검출 기법과 **퓨어 퍼슈트(Pure Pursuit)** 알고리즘을 결합하는 프로젝트입니다. 이를 통해 실시간으로 조향각을 산출하고, 경로함수를 생성·시각화합니다.



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

## 환경 및 설치

```bash
# Python 3.7 이상
pip install opencv-python numpy matplotlib scipy
```

- `opencv-python`: 비디오 입출력, 픽셀 연산, 윈도우 그리기
- `numpy`: 배열·행렬 연산, 좌표 변환
- `matplotlib`: 실시간 플롯, FigureCanvas → OpenCV 호환 이미지 변환
- `scipy.optimize.fsolve`: 원 방정식과 다항식의 교점 계산

---

- `main.py` 내부 흐름:
  1. Matplotlib Figure/Axes 초기화
  2. 비디오 캡처 객체 생성 및 확인
  3. `while` 루프: 프레임 읽기 → 각종 처리 함수 호출 → 플롯 갱신 → `imshow`
  4. 종료 조건 (`q` 입력)

---

## 주요 파라미터 및 설정

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

## 알고리즘 상세 설명

### 영상 전처리 (BEV → Gray → Blur → Binary)

```python
def Bev_Gray_Blurred_Binary_transform(frame):
    height, width = frame.shape[:2]
    src_pts = np.float32([[10, 450], [540, 430], [435, 240], [190, 240]])
    dst_pts = np.float32([[0, height], [width, height], [width, 0], [0, 0]])
    bev_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    bev_frame = cv2.warpPerspective(frame, bev_matrix, (width, height)) # bev
    gray_frame = cv2.cvtColor(bev_frame, cv2.COLOR_BGR2GRAY) # 흑백 변환
    blurred_frame = cv2.GaussianBlur(gray_frame, (19, 19), 0) # 가우시안 블러
    ret, binary_frame = cv2.threshold(blurred_frame, 225, 255, cv2.THRESH_BINARY) # 이진화 적용
    return  binary_frame
```


### 히스토그램 기반 초기 차선 위치
```python
hist = np.sum(binary[int(1.15*h/3):,:],axis=0)
mid = hist.shape[0]//2
L_idx = np.argmax(hist[:mid])
R_idx = np.argmax(hist[mid:])+mid
```
- 프레임 하단 2/3 영역에서 흰 픽셀 열 합산
- 좌/우 절반 구간 최댓값으로 초기 x


### 슬라이딩 윈도우 차선 추적
```python
def window_info(frame,num_win,win,prev_x,margin):
    h = frame.shape[0]//num_win
    y_low = frame.shape[0]-(win+1)*h
    y_high= frame.shape[0]-win*h
    x_low = max(0,prev_x-margin)
    x_high= min(frame.shape[1],prev_x+margin)
    indices = np.column_stack(np.where(frame[y_low:y_high,x_low:x_high]>0))
    if len(indices)<min_pts: return None
    x_mean = x_low + indices[:,1].mean()
    y_mean = y_low + indices[:,0].mean()
    return int(x_mean),int(y_mean)
```
- 각 윈도우 높이 = `frame.height/num_win`
- `prev_x` 기준 좌우 `margin` 범위 검색
- 흰 픽셀 위치 인덱스 → 평균으로 윈도우 중심 산출
- 픽셀 수 부족 시 해당 프레임 탐색 종료


### 픽셀 → 미터 좌표 변환
```python
def pixel_to_meter(x_px,y_px):
    x_m = (x_px-origin_x)*S_x
    y_m = (origin_y-y_px)*S_y
    return y_m, -x_m
```
- x: 오른쪽 (+), y: 전방 (+)로 좌표계 재정의
- 차량 후륜축 중심을 (0,0) 기준으로 보정


### 다항식 피팅
```python
# Right lane (3차)
R_x,R_y = zip(*R_pts)
R_coeff = np.polyfit(R_x,R_y,3)
R_func  = np.poly1d(R_coeff)
# Left lane (1차)
L_x,L_y = zip(*L_pts)
L_coeff = np.polyfit(L_x,L_y,1)
L_func  = np.poly1d(L_coeff)
```


### 룩어헤드 포인트 계산
```python
def lookahead(a,b,r,func,nv_pts):
    def eq(x): return (x-a)**2+(func(x)-b)**2-r**2
    guesses = [p[0] for p in nv_pts]
    sol = fsolve(eq,guesses)
    pts = [(x,func(x)) for x in sol if x>0]
    return pts
```
- 차량 위치 `(a,b)=(0,0)`에서 반지름 `r= Ld` 원 방정식과 nv_path 함수의 교점 풀기
- 초기 추정치로 nv_pts x좌표 리스트 사용 → 빠른 수렴 유도


### Pure Pursuit 조향각 계산
```python
v_car = np.array([1,0])
v_look= np.array([x-0,y-0])
α = acos(dot(v_car,v_look)/(||v_car||·||v_look||))
# y<0 시 음수 부호 부여
δ = atan(2*L*sin(α)/Ld)
```
- **α (라디안)**: 차량 진행축과 룩어헤드 벡터 사이 각
- **δ (조향각)**: Pure Pursuit 공식에 따라 계산 (휠베이스 `L` 포함)

---

