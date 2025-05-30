##  https://hackmd.io/@ZiyanGZiyaNG/AI_Reflex  HackMD網站
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random
import time
from collections import deque, Counter

player = ['Rock', 'Paper', 'Scissors']
num_classes = len(player)
player_score = 0
ai_score = 0
round_count = 0

training_data = []
training_labels = []
training_phase = True
training_round = 0
difficulty = None

gesture_order = ['Rock'] * 5 + ['Paper'] * 5 + ['Scissors'] * 5
label_to_index = {'Rock': 0, 'Paper': 1, 'Scissors': 2}

cap = cv2.VideoCapture(0)

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return np.array([area, perimeter, w/h])
    return np.array([0, 0, 1])

def train_model():
    if len(training_data) > 0:
        X = np.array(training_data)
        y = np.array(training_labels)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        return clf
    return None

player_memory = deque(maxlen=10)

def predict_next_move():
    if len(player_memory) < 3:
        return random.randint(0, num_classes - 1)
    most_common = Counter(player_memory).most_common(1)[0][0]
    return (most_common + 1) % num_classes

def ai_choice(difficulty, last_player_choice=None):
    if difficulty == 0:
        return random.randint(0, num_classes - 1)
    elif difficulty == 1:
        if last_player_choice is not None and random.random() < 0.5:
            return (last_player_choice + 1) % num_classes
        return random.randint(0, num_classes - 1)
    else:
        if random.random() < 0.8:
            return predict_next_move()
        return random.randint(0, num_classes - 1)

def determine_winner(player_idx, ai_idx):
    global player_score, ai_score
    if player_idx == ai_idx:
        return "Tie!"
    if (player_idx == 0 and ai_idx == 2) or (player_idx == 1 and ai_idx == 0) or (player_idx == 2 and ai_idx == 1):
        player_score += 1
        return "U win!"
    ai_score += 1
    return "AI win!"

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

countdown = 0
last_prediction_time = 0
current_player_choice = None
current_ai_choice = None
current_result = ""
clf = None
no_gesture_detected = False
last_player_choice = None

cv2.namedWindow('Rock Paper Scissors')
cv2.setMouseCallback('Rock Paper Scissors', mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    roi = frame[int(height / 4):int(3 * height / 4), int(width/4):int(3 * width / 4)]
    current_time = time.time()

    if difficulty is None:
        buttons, btn_width, btn_height, start_y, btn_spacing = draw_difficulty_ui(frame, -1, -1)
        cv2.setMouseCallback('Rock Paper Scissors', mouse_callback, (buttons, btn_width, btn_height, start_y, btn_spacing))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('0'):
            difficulty = 0
        elif key == ord('1'):
            difficulty = 1
        elif key == ord('2'):
            difficulty = 2
    else:
        if countdown == 0:
            countdown = 10
            last_prediction_time = current_time
            current_player_choice = None
            current_ai_choice = None
            current_result = ""
            no_gesture_detected = False

        remaining_time = int(10 - (current_time - last_prediction_time))
        if remaining_time != countdown:
            countdown = max(0, remaining_time)

        if countdown == 0 and current_player_choice is None:
            features = extract_features(roi)
            if np.all(features == [0, 0, 1]):
                no_gesture_detected = True
                break
            else:
                if training_phase:
                    expected_label = gesture_order[training_round]
                    training_data.append(features)
                    training_labels.append(label_to_index[expected_label])
                    training_round += 1
                    if training_round >= len(gesture_order):
                        training_phase = False
                        clf = train_model()
                        round_count = 0
                        player_score = 0
                        ai_score = 0
                else:
                    prediction = clf.predict([features])[0]
                    current_player_choice = player[prediction]
                    player_memory.append(prediction)
                    ai_idx = ai_choice(difficulty, last_player_choice)
                    current_ai_choice = player[ai_idx]
                    current_result = determine_winner(prediction, ai_idx)
                    round_count += 1
                    last_player_choice = prediction

        draw_game_ui(frame, countdown, current_player_choice, current_ai_choice, current_result, training_phase, difficulty)
        cv2.rectangle(frame, (int(width/4), int(height/4)),
                      (int(3*width/4), int(3*height/4)), (0, 0, 255), 2)
        cv2.putText(frame, "Hand in red box (q leaves)", (10, height-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Rock Paper Scissors', frame)
    if difficulty is not None and cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if no_gesture_detected:
        print("NO SIGN!! BREAK!!")
        break

cap.release()
cv2.destroyAllWindows()