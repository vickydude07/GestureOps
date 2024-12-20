import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
import random
import time
import threading
import collections
import platform
import sys
import pyautogui
import pickle
import openai  # Import OpenAI for AI integration

# Set pyautogui settings for smoother movement
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# Initialize Mediapipe and other global variables
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Process only one hand
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mpDraw = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

active_mode = None

# Drawing mode variables
ml = 150
max_x, max_y = 250 + ml, 50
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0, 0
mask = None  # We'll initialize the mask when we know the frame size
xii, yii = 0, 0
ctime, ptime = 0, 0

# Load tools image
tools = cv2.imread("tools.png")
if tools is None:
    print("Error: tools.png image not found.")
    tools = np.zeros((50, 250, 3), dtype=np.uint8)
tools = tools.astype("uint8")

# Load the trained model
try:
    model_dict = pickle.load(open("./model.p", "rb"))
    model = model_dict["model"]
except FileNotFoundError:
    print(
        "Error: model.p file not found. Please ensure the model file is in the same directory."
    )
    model = None

# Labels dictionary for hand signs
labels_dict = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
    26: "0",
    27: "1",
    28: "2",
    29: "3",
    30: "4",
    31: "5",
    32: "6",
    33: "7",
    34: "8",
    35: "9",
    36: "capslock",
    37: "space",
    38: "backspace",
}
# Initialize typed text buffer for AI integration
typed_text = ""

# Counter for characters since last AI suggestion
chars_since_last_suggestion = 0

# Flag to indicate if AI suggestion is in progress
ai_suggestion_in_progress = False


def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(np.degrees(radians))
    return angle


def get_distance(landmark_list):
    if len(landmark_list) < 2:
        return
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    L = np.hypot(x2 - x1, y2 - y1)
    return np.interp(L, [0, 1], [0, 1000])


def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        index_finger_tip = hand_landmarks.landmark[
            mpHands.HandLandmark.INDEX_FINGER_TIP
        ]
        return index_finger_tip
    return None


# Use a deque to store recent positions for smoothing
position_history = collections.deque(maxlen=10)
smoothed_position = None
alpha = 0.2  # Smoothing factor between 0 and 1


def move_mouse(index_finger_tip):
    global smoothed_position
    if index_finger_tip is not None:
        x = index_finger_tip.x
        y = index_finger_tip.y

        # Map to screen coordinates
        screen_x = np.interp(x, [0, 1], [0, screen_width])
        screen_y = np.interp(y, [0, 1], [0, screen_height])

        # Initialize smoothed_position if None
        if smoothed_position is None:
            smoothed_position = (screen_x, screen_y)
        else:
            # Apply exponential smoothing
            smoothed_x = alpha * screen_x + (1 - alpha) * smoothed_position[0]
            smoothed_y = alpha * screen_y + (1 - alpha) * smoothed_position[1]
            smoothed_position = (smoothed_x, smoothed_y)

        avg_x = int(smoothed_position[0])
        avg_y = int(smoothed_position[1])

        pyautogui.moveTo(avg_x, avg_y)


def is_left_click(landmark_list, thumb_index_dist):
    return (
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50
        and get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90
        and thumb_index_dist > 50
    )


def is_right_click(landmark_list, thumb_index_dist):
    return (
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50
        and get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90
        and thumb_index_dist > 50
    )


def is_double_click(landmark_list, thumb_index_dist):
    return (
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50
        and get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50
        and thumb_index_dist > 50
    )


def is_screenshot(landmark_list, thumb_index_dist):
    return (
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50
        and get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50
        and thumb_index_dist < 50
    )


# Add a timestamp to limit click actions
last_click_time = time.time()
last_keypress_time = time.time()  # For hand sign recognition


def detect_gesture(frame, landmark_list, processed):
    global last_click_time
    if len(landmark_list) >= 21:
        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = get_distance([landmark_list[4], landmark_list[5]])

        # Move mouse
        move_mouse(index_finger_tip)

        # Limit the frequency of click actions
        current_time = time.time()
        if current_time - last_click_time > 1:
            if is_left_click(landmark_list, thumb_index_dist):
                pyautogui.click(button="left")
                last_click_time = current_time
            elif is_right_click(landmark_list, thumb_index_dist):
                pyautogui.click(button="right")
                last_click_time = current_time
            elif is_double_click(landmark_list, thumb_index_dist):
                pyautogui.doubleClick()
                last_click_time = current_time
            elif is_screenshot(landmark_list, thumb_index_dist):
                im1 = pyautogui.screenshot()
                label = random.randint(1, 1000)
                im1.save(f"my_screenshot_{label}.png")
                last_click_time = current_time


def detect_cameras(max_cameras=5):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras


class CameraSelectionDialog(tk.Toplevel):
    def __init__(self, parent, cameras):
        super().__init__(parent)
        self.title("Select Camera")
        self.cameras = cameras
        self.selected_camera = None
        self.geometry("300x150")
        label = tk.Label(self, text="Select a camera:")
        label.pack(pady=10)
        self.var = tk.IntVar()
        for cam in cameras:
            rb = tk.Radiobutton(
                self, text=f"Camera {cam}", variable=self.var, value=cam
            )
            rb.pack(anchor="w")
        button = tk.Button(self, text="OK", command=self.on_ok)
        button.pack(pady=10)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)

    def on_ok(self):
        self.selected_camera = self.var.get()
        self.destroy()

    def on_close(self):
        self.destroy()


def select_camera():
    cameras = detect_cameras()
    if not cameras:
        messagebox.showerror("Error", "No cameras found.")
        root.destroy()
        return None
    dialog = CameraSelectionDialog(root, cameras)
    if dialog.selected_camera is not None:
        cap = cv2.VideoCapture(dialog.selected_camera)
        if not cap.isOpened():
            messagebox.showerror(
                "Error", f"Cannot open camera {dialog.selected_camera}"
            )
            root.destroy()
            return None
        else:
            return cap
    else:
        root.destroy()
        return None


def getTool(x, frame_width):
    ml_resized = int(ml * frame_width / 640)
    if x < int(50 * frame_width / 640) + ml_resized:
        return "line"
    elif x < int(100 * frame_width / 640) + ml_resized:
        return "rectangle"
    elif x < int(150 * frame_width / 640) + ml_resized:
        return "draw"
    elif x < int(200 * frame_width / 640) + ml_resized:
        return "circle"
    else:
        return "erase"


def index_raised(yi, y9):
    if (y9 - yi) > 40:
        return True
    return False


def give_suggestion(pre):
    openai.api_key = "ollama"
    openai.base_url = "http://localhost:11434/v1/"
    response = openai.chat.completions.create(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that takes a sentence and fills it appropriately. Only suggest one word next to the sentence or fix the current word. Provide the corrected sentence. If you could not give a suggestion, return the original sentence. **DO NOT SAY OR MENTION ANYTHING ELSE APART FROM THIS*",
            },
            {"role": "user", "content": pre},
        ],
    )
    # Extract the assistant's reply
    suggestion = response.choices[0].message.content.strip()
    return suggestion


def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frameRGB.shape[:2]

        global mask
        global ai_suggestion_in_progress
        if mask is None or mask.shape[:2] != frameRGB.shape[:2]:
            mask = np.ones(frameRGB.shape[:2], dtype="uint8") * 255

        if ai_suggestion_in_progress:
            # Display "AI Suggesting..." message
            display_frame = np.full_like(frameRGB, 255)  # White background
            cv2.putText(
                display_frame,
                "AI Suggesting...",
                (width // 2 - 100, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            frameRGB = display_frame
        else:
            if active_mode == "mouse_control":
                processed = hands.process(frameRGB)

                if processed.multi_hand_landmarks:
                    hand_landmarks = processed.multi_hand_landmarks[0]
                    mpDraw.draw_landmarks(
                        frameRGB, hand_landmarks, mpHands.HAND_CONNECTIONS
                    )
                    landmark_list = []
                    for lm in hand_landmarks.landmark:
                        landmark_list.append((lm.x, lm.y))

                    detect_gesture(frameRGB, landmark_list, processed)

            elif active_mode == "drawing":
                global curr_tool, time_init, rad, var_inits, prevx, prevy, xii, yii, ctime, ptime
                processed = hands.process(frameRGB)

                # Create a white background image for display
                display_frame = np.full_like(frameRGB, 255)  # White background

                if processed.multi_hand_landmarks:
                    for i in processed.multi_hand_landmarks:
                        # Perform hand detection on frameRGB but draw on display_frame
                        mpDraw.draw_landmarks(
                            display_frame, i, mpHands.HAND_CONNECTIONS
                        )

                        x = int(i.landmark[8].x * width)
                        y = int(i.landmark[8].y * height)

                        ml_resized = int(ml * width / 640)
                        max_x_resized = int(max_x * width / 640)
                        max_y_resized = int(max_y * height / 480)

                        if x < max_x_resized and y < max_y_resized and x > ml_resized:
                            if time_init:
                                ctime = time.time()
                                time_init = False
                            ptime = time.time()
                            cv2.circle(
                                display_frame, (x, y), int(rad), (0, 255, 255), 2
                            )
                            rad -= 1
                            if (ptime - ctime) > 0.8:
                                curr_tool = getTool(x, width)
                                print("Your current tool set to:", curr_tool)
                                time_init = True
                                rad = 40
                        else:
                            time_init = True
                            rad = 40

                        xi = int(i.landmark[12].x * width)
                        yi = int(i.landmark[12].y * height)
                        y9 = int(i.landmark[9].y * height)

                        if curr_tool == "draw":
                            if index_raised(yi, y9):
                                cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
                                prevx, prevy = x, y
                            else:
                                prevx = x
                                prevy = y

                        elif curr_tool == "line":
                            if index_raised(yi, y9):
                                if not var_inits:
                                    xii, yii = x, y
                                    var_inits = True
                                cv2.line(
                                    display_frame,
                                    (xii, yii),
                                    (x, y),
                                    (50, 152, 255),
                                    thick,
                                )
                            else:
                                if var_inits:
                                    cv2.line(mask, (xii, yii), (x, y), 0, thick)
                                    var_inits = False

                        elif curr_tool == "rectangle":
                            if index_raised(yi, y9):
                                if not var_inits:
                                    xii, yii = x, y
                                    var_inits = True
                                cv2.rectangle(
                                    display_frame,
                                    (xii, yii),
                                    (x, y),
                                    (0, 255, 255),
                                    thick,
                                )
                            else:
                                if var_inits:
                                    cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
                                    var_inits = False

                        elif curr_tool == "circle":
                            if index_raised(yi, y9):
                                if not var_inits:
                                    xii, yii = x, y
                                    var_inits = True
                                radius = int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5)
                                cv2.circle(
                                    display_frame,
                                    (xii, yii),
                                    radius,
                                    (255, 255, 0),
                                    thick,
                                )
                            else:
                                if var_inits:
                                    radius = int(
                                        ((xii - x) ** 2 + (yii - y) ** 2) ** 0.5
                                    )
                                    cv2.circle(mask, (xii, yii), radius, 0, thick)
                                    var_inits = False

                        elif curr_tool == "erase":
                            if index_raised(yi, y9):
                                cv2.circle(display_frame, (x, y), 30, (0, 0, 0), -1)
                                cv2.circle(mask, (x, y), 30, 255, -1)

                # Apply the mask to the display_frame
                op = cv2.bitwise_and(
                    display_frame, display_frame, mask=mask.astype("uint8")
                )
                display_frame = op

                tools_resized_width = int((max_x - ml) * width / 640)
                tools_resized_height = int(max_y * height / 480)
                tools_resized = cv2.resize(
                    tools, (tools_resized_width, tools_resized_height)
                )

                ml_resized = int(ml * width / 640)
                display_frame[
                    :tools_resized_height,
                    ml_resized : ml_resized + tools_resized_width,
                ] = cv2.addWeighted(
                    tools_resized,
                    0.7,
                    display_frame[
                        :tools_resized_height,
                        ml_resized : ml_resized + tools_resized_width,
                    ],
                    0.3,
                    0,
                )

                cv2.putText(
                    display_frame,
                    curr_tool,
                    (int((270 + ml) * width / 640), 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

                # Set frameRGB to display_frame for display
                frameRGB = display_frame

            elif active_mode == "hand_signs" and model is not None:
                global last_keypress_time, typed_text, chars_since_last_suggestion
                processed = hands.process(frameRGB)

                if processed.multi_hand_landmarks:
                    hand_landmarks = processed.multi_hand_landmarks[0]
                    mpDraw.draw_landmarks(
                        frameRGB, hand_landmarks, mpHands.HAND_CONNECTIONS
                    )

                    data_aux = []
                    x_ = []
                    y_ = []

                    for lm in hand_landmarks.landmark:
                        x = lm.x
                        y = lm.y
                        x_.append(x)
                        y_.append(y)

                    for lm in hand_landmarks.landmark:
                        x = lm.x
                        y = lm.y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    # Ensure data_aux has the correct size
                    if len(data_aux) == model.n_features_in_:
                        prediction = model.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict[int(prediction[0])]

                        x1 = int(min(x_) * width) - 10
                        y1 = int(min(y_) * height) - 10
                        x2 = int(max(x_) * width) + 10
                        y2 = int(max(y_) * height) + 10

                        cv2.rectangle(frameRGB, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(
                            frameRGB,
                            predicted_character,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.3,
                            (0, 0, 0),
                            3,
                            cv2.LINE_AA,
                        )

                        # Trigger key press every 1 second
                        current_time = time.time()
                        if current_time - last_keypress_time > 1.0:
                            try:
                                pyautogui.press(predicted_character.lower())
                                print(f"Pressed key: {predicted_character.lower()}")
                                # Append to typed text
                                if predicted_character.lower() not in [
                                    "backspace",
                                    "capslock",
                                    "space",
                                ]:
                                    typed_text += predicted_character.lower()
                                chars_since_last_suggestion += 1
                                print(
                                    f"Typed text: {typed_text}, Chars since last suggestion: {chars_since_last_suggestion}"
                                )

                                # Check if we have typed 3 new characters
                                if chars_since_last_suggestion >= 3:
                                    # Start AI suggestion in a separate thread
                                    ai_suggestion_in_progress = True
                                    threading.Thread(
                                        target=process_ai_suggestion, args=(typed_text,)
                                    ).start()
                                    chars_since_last_suggestion = 0  # Reset counter
                            except Exception as e:
                                print(f"Error pressing key: {e}")
                            last_keypress_time = current_time

                    else:
                        print(
                            f"Feature size mismatch: expected {model.n_features_in_}, got {len(data_aux)}"
                        )

        label_width = video_label.winfo_width()
        label_height = video_label.winfo_height()
        if label_width > 0 and label_height > 0:
            frameRGB_resized = cv2.resize(frameRGB, (label_width, label_height))
            img = Image.fromarray(frameRGB_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
    video_label.after(10, update_frame)


def process_ai_suggestion(text):
    global ai_suggestion_in_progress, typed_text
    try:
        suggestion = give_suggestion(text)
        print(f"AI Suggestion: {suggestion}")

        # Simulate deleting the current text
        for _ in range(len(typed_text)):
            pyautogui.press("backspace")

        # Type the new suggestion
        pyautogui.write(suggestion)
        # Update typed_text
        typed_text = suggestion

    except Exception as e:
        print(f"Error during AI suggestion: {e}")
    finally:
        ai_suggestion_in_progress = False


def set_mode(mode):
    global active_mode, mask, curr_tool, time_init, rad, var_inits, prevx, prevy, xii, yii, ctime, ptime, typed_text, chars_since_last_suggestion
    active_mode = mode

    if mode == "drawing":
        # Reset variables for drawing mode
        curr_tool = "select tool"
        time_init = True
        rad = 40
        var_inits = False
        prevx, prevy = 0, 0
        xii, yii = 0, 0
        ctime, ptime = 0, 0
        # Reset the mask
        mask = None
    elif mode == "hand_signs":
        # Reset typed text and character counter
        typed_text = ""
        chars_since_last_suggestion = 0


root = tk.Tk()
root.title("Hand Gesture UI")

column_width = 150
root.grid_columnconfigure(0, minsize=column_width)
root.grid_columnconfigure(1, weight=1)

for i in range(4):
    root.grid_rowconfigure(i, minsize=column_width)

button1 = tk.Button(
    root, text="Mouse Control", command=lambda: set_mode("mouse_control")
)
button1.grid(row=0, column=0, sticky="nsew")

button2 = tk.Button(root, text="Drawing Mode", command=lambda: set_mode("drawing"))
button2.grid(row=1, column=0, sticky="nsew")

button3 = tk.Button(
    root, text="Hand Sign Recognition", command=lambda: set_mode("hand_signs")
)
button3.grid(row=2, column=0, sticky="nsew")

button4 = tk.Button(root, text="Stop", command=lambda: set_mode(None))
button4.grid(row=3, column=0, sticky="nsew")

video_frame = tk.Frame(root)
video_frame.grid(row=0, column=1, rowspan=4, sticky="nsew")
video_frame.grid_rowconfigure(0, weight=1)
video_frame.grid_columnconfigure(0, weight=1)
video_frame.grid_propagate(False)

video_label = tk.Label(video_frame)
video_label.grid(row=0, column=0, sticky="nsew")

cap = cv2.VideoCapture(0)

if cap:
    update_frame()
    root.mainloop()
