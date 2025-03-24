import cv2
import mediapipe as mp
import numpy as np
import pulsectl
import sounddevice as sd
from scipy.io.wavfile import read

# ------------------------------
# INITIALIZATION
# ------------------------------

# Initialize MediaPipe Hands for hand detection (allow up to 2 hands)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Initialize pulsectl for system volume control (PulseAudio)
pulse = pulsectl.Pulse('airtune')
sinks = pulse.sink_list()
sink = sinks[0] if sinks else None

def set_volume(vol_fraction):
    """
    Set the system volume using pulsectl.
    vol_fraction: A float between 0.0 (mute) and 1.0 (max volume)
    """
    if sink is not None:
        pulse.volume_set_all_chans(sink, vol_fraction)

# Load the sample audio file
sample_rate, audio_data = read("sample.wav")

def play_audio():
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()

# ------------------------------
# INITIAL STORED VALUES
# ------------------------------
# Start with default volume of 50% and pitch 1.0 (neutral)
current_volume_fraction = 0.5  # fraction from 0.0 to 1.0
current_volume_percent = 50
current_pitch = 1.0

# Apply initial volume setting
set_volume(current_volume_fraction)

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------

def calculate_distance(pt1, pt2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def adjust_pitch(pitch_value):
    """
    Placeholder for pitch adjustment.
    Replace with your audio processing logic if needed.
    """
    print(f"Adjusted pitch value: {pitch_value}")

# ------------------------------
# MAIN LOOP: PROCESS FRAMES
# ------------------------------

cap = cv2.VideoCapture(0)
play_audio()

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip image horizontally for a mirror view
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    # Convert image to RGB and process with MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Flags to check if we update volume and pitch in this frame
    update_volume = False
    update_pitch = False

    if results.multi_hand_landmarks:
        hand_landmarks_all = results.multi_hand_landmarks

        # ----- Volume Control using the first hand -----
        hand1 = hand_landmarks_all[0]
        mp_draw.draw_landmarks(img, hand1, mp_hands.HAND_CONNECTIONS)

        lm_list1 = []
        for lm in hand1.landmark:
            lm_list1.append((int(lm.x * w), int(lm.y * h)))

        if len(lm_list1) >= 9:
            thumb_tip = lm_list1[mp_hands.HandLandmark.THUMB_TIP.value]
            index_tip = lm_list1[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
            distance_vol = calculate_distance(thumb_tip, index_tip)

            # Map the distance to a volume fraction (adjust ranges as needed)
            new_vol_fraction = np.interp(distance_vol, [30, 200], [0.0, 1.0])
            current_volume_fraction = new_vol_fraction
            current_volume_percent = np.interp(distance_vol, [30, 200], [0, 100])
            set_volume(current_volume_fraction)
            update_volume = True

            # Visualize volume control
            cv2.putText(img, f'Volume: {int(current_volume_percent)}%', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.line(img, thumb_tip, index_tip, (255, 0, 0), 3)
            cv2.circle(img, thumb_tip, 8, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, index_tip, 8, (0, 255, 0), cv2.FILLED)

        # ----- Pitch Control using two hands -----
        if len(hand_landmarks_all) >= 2:
            hand2 = hand_landmarks_all[1]
            mp_draw.draw_landmarks(img, hand2, mp_hands.HAND_CONNECTIONS)
            lm_list2 = []
            for lm in hand2.landmark:
                lm_list2.append((int(lm.x * w), int(lm.y * h)))

            if len(lm_list2) >= 9:
                index_tip_hand1 = lm_list1[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
                index_tip_hand2 = lm_list2[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
                distance_pitch = calculate_distance(index_tip_hand1, index_tip_hand2)
                # Map the distance to a pitch value (example range: 0.8 to 1.2)
                new_pitch = np.interp(distance_pitch, [50, 300], [0.8, 1.2])
                current_pitch = new_pitch
                adjust_pitch(current_pitch)
                update_pitch = True

                cv2.putText(img, f'Pitch: {current_pitch:.2f}', (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.line(img, index_tip_hand1, index_tip_hand2, (0, 0, 255), 3)
                cv2.circle(img, index_tip_hand1, 8, (0, 255, 255), cv2.FILLED)
                cv2.circle(img, index_tip_hand2, 8, (0, 255, 255), cv2.FILLED)

    # Show the video feed
    cv2.imshow('AirTune: Hands-Free Audio Control', img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------
# CLEANUP
# ------------------------------

cap.release()
cv2.destroyAllWindows()
pulse.close()

