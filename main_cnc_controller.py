import cv2
import mediapipe as mp
import numpy as np
import serial
import time
 
ser = serial.Serial('/dev/ttyACM0', 115200)   
time.sleep(2)   

# Wake up GRBL
ser.write(b"\r\n\r\n")
time.sleep(2)
ser.flushInput()

# Initialize MediaPipe Hand Detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8)

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)
ws, hs = 1280, 720   
cap.set(3, ws)
cap.set(4, hs)

# Function to send G-code commands to GRBL
def send_gcode(command):
    command += '\n'
    ser.write(command.encode())
    time.sleep(0.05)
    response = ser.readline().decode().strip()
    print(f"Sent: {command.strip()}, Response: {response}")

# Variables to track pinch state
pinch_detected = False

# Main Loop to Track Finger and Move Stepper Motors
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)   
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get index finger tip position (landmark 8) and thumb tip position (landmark 4)
            index_finger = hand_landmarks.landmark[8]
            thumb = hand_landmarks.landmark[4]

            # Convert normalized hand landmark positions to screen coordinates
            finger_x = int(index_finger.x * ws)
            finger_y = int(index_finger.y * hs)
            thumb_x = int(thumb.x * ws)
            thumb_y = int(thumb.y * hs)

            # Display finger position
            cv2.circle(img, (finger_x, finger_y), 10, (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'X: {finger_x}, Y: {finger_y}', (finger_x + 10, finger_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check for pinch gesture (index finger and thumb close to each other)
            distance = np.hypot(finger_x - thumb_x, finger_y - thumb_y)
            if distance < 40:
                if not pinch_detected:
                    print("Pinch detected! Moving servo to 90 degrees.")
                    send_gcode("M3 S30")  
                    pinch_detected = True
            else:
                if pinch_detected:
                    print("Pinch released! Moving servo to 0 degrees.")
                    send_gcode("M3 S90")  
                    pinch_detected = False

            # Generate G-code based on finger position
            
            cnc_x = np.interp(finger_x, [0, ws], [0, 100])   
            cnc_y = np.interp(finger_y, [0, hs], [0, 100])  
            feed_rate = 2000   

            # Send the G-code command to GRBL
            gcode_command = f"G01 X{cnc_x:.2f} Y{cnc_y:.2f} F{feed_rate}"
            send_gcode(gcode_command)

            # Draw landmarks
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the OpenCV image with finger tracking
    cv2.imshow("Hand Tracking", img)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
ser.close()
