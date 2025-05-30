![c6ce8d0b9a7b28f9c2dee8171da98b8f ](https://hackmd.io/_uploads/rkJ--Z1xex.jpg)


# 簡介
這是結合openCV與機器學習的猜拳遊戲，透過鏡頭辨識玩家手勢，並使用隨機森林模型進行判斷與對戰。系統包含訓練與對戰兩階段，支援三種難度，提供即時互動與AI對戰體驗。
# 需下載的東東
* 下載
```
!pip install opencv-python # 下載opencv
!pip install numpy # 下載numpy
!pip install scikit-learn # 下載sklearn
```
* IDE(Win跟Mac) 版：
    * 打開IDE，找到terminal把下面的東西塞進去
    ![螢幕擷取畫面 2025-04-30 004410](https://hackmd.io/_uploads/HJcfMYC1ee.png)
* Win 10/11 版：
    * win + R 打開執行，輸入cmd或powershell(都一樣啦)，一樣把下面的東西塞進去
    ![螢幕擷取畫面 2025-04-30 004525](https://hackmd.io/_uploads/SyYEGK0Jxe.png) 
    或
    ![螢幕擷取畫面 2025-04-30 004556](https://hackmd.io/_uploads/S1yLfK01ee.png)
    打開長這樣
    ![螢幕擷取畫面 2025-04-30 004628](https://hackmd.io/_uploads/BJGnzY0ygx.png)

* Mac 版：
    * Command + space，繼續把下面的東西塞進去
   *~~窩與不知道，沒錢買Macbook~~*
# 初始化+模組導入
```
import cv2  # 影像+UI
import numpy as np  # 數值+陣列
from sklearn.ensemble import RandomForestClassifier  # 隨機森林
import random  # 隨機
import time  # 倒數計時
from collections import deque, Counter  # deque 記憶玩家手勢，Counter 出現次數

# 玩家手勢清單
player = ['Rock', 'Paper', 'Scissors']
num_classes = len(player)  # 三種手勢的總數
player_score = 0  # 玩家分數
ai_score = 0  # AI 分數
round_count = 0  # 對戰輪數

# 訓練資料變數
training_data = []  # 儲存擷取的特徵向量
training_labels = []  # 對應手勢標籤
training_phase = True  # 是否為訓練階段
training_round = 0  # 訓練第幾輪
difficulty = None  # 難度尚未選擇

# 指定訓練的手勢順序（每種手勢各5次）
gesture_order = ['Rock'] * 5 + ['Paper'] * 5 + ['Scissors'] * 5
label_to_index = {'Rock': 0, 'Paper': 1, 'Scissors': 2}  # 手勢對應的索引

# 開啟攝影機
cap = cv2.VideoCapture(0)
```
# 特徵get與模型訓練
```
# 擷取紅框中的影像特徵
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 轉灰階
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 二值化 (0, 1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找輪廓
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)  # 找最大輪廓
        area = cv2.contourArea(largest_contour)  # 面積
        perimeter = cv2.arcLength(largest_contour, True)  # 周長
        x, y, w, h = cv2.boundingRect(largest_contour)  # 外接矩形
        return np.array([area, perimeter, w/h])  # 回傳特徵向量
    return np.array([0, 0, 1])  # 若沒找到輪廓，回傳預設值

# 訓練模型
def train_model():
    if len(training_data) > 0:
        X = np.array(training_data)
        y = np.array(training_labels)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)  # 建立模型
        clf.fit(X, y)  # 訓練
        return clf
    return None
```
#  AI出招策略
```
# 紀錄玩家歷史手勢，最多記 10 次
player_memory = deque(maxlen=10)

# Hard
def predict_next_move():
    if len(player_memory) < 3:
        return random.randint(0, num_classes - 1)  # 資料不足，隨機猜
    most_common = Counter(player_memory).most_common(1)[0][0]  # 找出最常出現的手勢
    return (most_common + 1) % num_classes  # 出剋制那招的手勢

# 根據難度決定 AI 出招
def ai_choice(difficulty, last_player_choice=None):
    if difficulty == 0:  # Easy 100% 隨機
        return random.randint(0, num_classes - 1)
    elif difficulty == 1:  # Normal：50% 模仿、50% 隨機
        if last_player_choice is not None and random.random() < 0.5:
            return (last_player_choice + 1) % num_classes
        return random.randint(0, num_classes - 1)
    else:  # Hard：80% 預測、20% 隨機
        if random.random() < 0.8:
            return predict_next_move()
        return random.randint(0, num_classes - 1)
```
# 勝負+UI
```
# 判定勝負與加分
def determine_winner(player_idx, ai_idx):
    global player_score, ai_score
    if player_idx == ai_idx:
        return "Tie!"
    if (player_idx == 0 and ai_idx == 2) or (player_idx == 1 and ai_idx == 0) or (player_idx == 2 and ai_idx == 1):
        player_score += 1
        return "U win!"
    ai_score += 1
    return "AI win!"

# 選擇UI
def draw_difficulty_ui(frame, mouse_x, mouse_y):
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, height), (50, 50, 50), -1)
    cv2.putText(frame, "Choose difficulty", (width//2-100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    btn_width, btn_height = 290, 80
    btn_spacing = 50
    start_y = height//2 - 100
    buttons = [
        ("Easy (press 0)", (0, 255, 0)),
        ("Normal (press 1)", (0, 255, 255)),
        ("Hard (press 2)", (0, 0, 255))
    ]

    for i, (text, color) in enumerate(buttons):
        x1 = width//2 - btn_width//2
        y1 = start_y + i * (btn_height + btn_spacing)
        x2 = x1 + btn_width
        y2 = y1 + btn_height
        hover = x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2
        btn_color = (150, 150, 150) if hover else color
        cv2.rectangle(frame, (x1, y1), (x2, y2), btn_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, text, (x1+10, y1 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return buttons, btn_width, btn_height, start_y, btn_spacing

# 遊戲UI
def draw_game_ui(frame, countdown, player_choice, ai_choice, result, training_phase, difficulty):
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 100), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, height-50), (width, height), (50, 50, 50), -1)
    if training_phase:
        cv2.putText(frame, f"training phase: No {training_round+1}/15 bureau", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"please show: {gesture_order[training_round]}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        difficulty_text = ["Easy", "Normal", "Hard"][difficulty]
        cv2.putText(frame, f"player: {player_score}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"AI: {ai_score}", (width-100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"round: {round_count} (difficulty: {difficulty_text})", (width//50, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if countdown > 0:
        cv2.putText(frame, f"Countdown: {countdown}", (width//250, height-70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if player_choice and ai_choice and not training_phase:
        cv2.putText(frame, f"U: {player_choice}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"AI: {ai_choice}", (width-100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, result, (width//2-50, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 滑鼠選難度
def mouse_callback(event, x, y, flags, param):
    global difficulty
    if event == cv2.EVENT_LBUTTONDOWN and difficulty is None:
        buttons, btn_width, btn_height, start_y, btn_spacing = param
        for i in range(len(buttons)):
            x1 = width//2 - btn_width//2
            y1 = start_y + i * (btn_height + btn_spacing)
            x2 = x1 + btn_width
            y2 = y1 + btn_height
            if x1 <= x <= x2 and y1 <= y <= y2:
                difficulty = i
                break
```
# 主程式
```
# 滑鼠點擊
cv2.namedWindow("Rock Paper Scissors")
mouse_x, mouse_y = 0, 0
cv2.setMouseCallback("Rock Paper Scissors", mouse_callback,
                     draw_difficulty_ui(np.zeros((480, 640, 3), dtype=np.uint8), 0, 0))

last_time = time.time()  # 初始時間
countdown = 3  # 倒數秒數
player_choice = None  # 玩家本輪選擇
ai_move = None  # AI 本輪選擇
result = ""  # 本輪結果

while True:
    ret, frame = cap.read()  # 攝影機讀影像
    if not ret:
        break

    height, width = frame.shape[:2]  # 畫面尺寸
    roi = frame[100:400, 100:400]  # 定義紅框區域 (玩家出手區)
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)  # 繪製紅框

    key = cv2.waitKey(1) & 0xFF  # 讀取鍵盤輸入
    if key == 27:  # 按下 ESC 離開
        break
    elif key in [ord('0'), ord('1'), ord('2')]:  # 選擇難度
        difficulty = int(chr(key))

    # 如果難度尚未選擇，顯示難度選單
    if difficulty is None:
        draw_difficulty_ui(frame, mouse_x, mouse_y)
        cv2.imshow("Rock Paper Scissors", frame)
        continue

    # 倒數控制邏輯
    current_time = time.time()
    if current_time - last_time >= 1:  # 每秒倒數
        countdown -= 1
        last_time = current_time

        if countdown == 0:
            features = extract_features(roi)  # 擷取紅框特徵
            if training_phase:
                label = label_to_index[gesture_order[training_round]]  # 取得目前應該輸入的手勢標籤
                training_data.append(features)  # 加入訓練資料
                training_labels.append(label)
                training_round += 1
                if training_round >= len(gesture_order):  # 全部訓練完畢
                    clf = train_model()  # 訓練模型
                    training_phase = False  # 進入正式對戰階段
            else:
                if clf is not None:
                    player_prediction = clf.predict([features])[0]  # 預測玩家手勢
                    player_memory.append(player_prediction)  # 加入記憶
                    ai_idx = ai_choice(difficulty, player_prediction)  # AI 出招
                    player_choice = player[player_prediction]
                    ai_move = player[ai_idx]
                    result = determine_winner(player_prediction, ai_idx)  # 判定勝負
                    round_count += 1
            countdown = 4  # 重設倒數時間

    # 畫出當前畫面（遊戲階段 or 訓練階段）
    draw_game_ui(frame, countdown, player_choice, ai_move,
                 result, training_phase, difficulty)

    # 顯示畫面
    cv2.imshow("Rock Paper Scissors", frame)

# 結束後釋放資源
cap.release()
cv2.destroyAllWindows()
```
# 往後可優化的東西
沒有用DataBase，問就是怕爆
所以現在是把東西全部塞在Ram裡面，所以一定要釋放資源

# IDE

我只有用Vscode跟Pycharm，其他的窩不知道，但應該不會有問題(除非你用競程專用IDE E.g.CP editor)
![窩不知道](https://hackmd.io/_uploads/r1b5PKAkgx.jpg)

# 版本
查版本，cmd key
```
pip show 你想查的模組
```

我是用
Python **3.12.9** (不是Python2

OpenCV **4.11.0.86**

NumPY **1.26.4**

Sklearn **1.6.1**

######## 不要北七去亂下載其他version，應該不會有問題 ########
# 硬體
因為CPU的頻率和暫存記憶體會有差，所以我提供我的硬體，~~想炫耀~~
> CPU - Intel® Core™ i5-14500
> 
> MB - PRO B760M-A DDR4 II
> 
> RAM - DDR4 3200Hz 32GB
> 
> GPU - INNO3D GeForce RTX™ 4070 TWIN X2 OC WHITE


