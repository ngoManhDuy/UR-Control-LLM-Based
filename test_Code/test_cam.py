import cv2

# 1. Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 2. Open default camera (0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# 3. Processing loop
while True:
    ret, frame = cap.read()              # capture frame-by-frame :contentReference[oaicite:3]{index=3}
    if not ret:
        print("Error: Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale :contentReference[oaicite:4]{index=4}
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )  # detect faces :contentReference[oaicite:5]{index=5}

    # 4. Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0), 2
        )  # blue box, thickness=2 :contentReference[oaicite:6]{index=6}

    # 5. Display the result
    cv2.imshow('Face Tracker', frame)  # show annotated frame :contentReference[oaicite:7]{index=7}

    # 6. Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. Cleanup
cap.release()
cv2.destroyAllWindows()
