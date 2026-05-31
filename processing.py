from cv2 import VideoCapture
from cv2 import cvtColor as cv2CvtColor
from cv2 import COLOR_BGR2RGB as cv2COLOR_BGR2RGB
from mediapipe.python.solutions import hands, drawing_utils
from numpy import array, vstack, zeros
from json import load as jsonLoad



with open('config.json', 'r', encoding='utf-8') as file:
    config = jsonLoad(file)

with open(config['dataset']['info'], 'r', encoding='utf-8') as file:
    info = jsonLoad(file)



handsDynamic = hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    model_complexity = 0,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)



def isHandsInFrame(frame):
    result = handsDynamic.process(frame)
    
    if result.multi_hand_landmarks:
        return True
    
    return False

def normBuferFrames(arr):
    if arr.shape == (info['max_video_len'], 42, 3):
        return arr
    
    normBufer = zeros((info['max_video_len'], 42, 3))
    for i in range(len(arr)):
        normBufer[i] = arr[i]
    
    return normBufer

def processingFrame(frame, returnLMFrame: bool):
    result = handsDynamic.process(frame)
    pointsL = None
    pointsR = None

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            points = array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            points = points - points[0]

            if hand_info.classification[0].label == 'Left':
                pointsL = points.copy()
            else:
                pointsR = points.copy()
            
            if returnLMFrame:
                drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    hands.HAND_CONNECTIONS,
                    drawing_utils.DrawingSpec(color=(0, 0, 255)),
                    drawing_utils.DrawingSpec(color=(255, 0, 255))
                )
    
    if pointsL is None:
        pointsL = zeros((21, 3))

    if pointsR is None:
        pointsR = zeros((21, 3))
    
    if returnLMFrame:
        return frame, vstack((pointsL, pointsR))
    
    return vstack((pointsL, pointsR))

def processingVideo(name: str):
    cap = VideoCapture(name + '.mp4')
    allFrames = zeros((info['max_video_len'], 42, 3))

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2CvtColor(frame, cv2COLOR_BGR2RGB)
        tmp = processingFrame(frame=frame, returnLMFrame=False)
        allFrames[i] = tmp
        i += 1
    
    cap.release()
    return allFrames
