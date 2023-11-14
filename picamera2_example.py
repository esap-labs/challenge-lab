import cv2

from picamera2 import Picamera2


# Configure Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()


while True:
    # Capture frame
    frame = picam2.capture_array()

    # Display frame
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break