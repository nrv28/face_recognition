from flask import Flask, request,render_template,redirect
import numpy as np
import cv2

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('hey.html')


@app.route('/open_camera')
def open_camera():
        
    # Load the pre-trained face detector from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Sample names corresponding to detected faces
    person_names = ["Nirjay", "Nishant", "Harsh Maliya","Umar","Love Pathak"]

    # Load sample images for comparison
    known_images = [
        cv2.imread('person1.jpg'),
        cv2.imread('person2.jpg'),
        cv2.imread('person3.jpg'),
        cv2.imread('person4.jpg'),
        cv2.imread('person5.jpg')
    ]

    # Convert known images to grayscale
    known_images_gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in known_images]

    # Open a connection to the webcam (0 represents the default camera)
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Iterate through detected faces
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangles around the detected faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Crop the face region for comparison
            face_roi = gray[y:y+h, x:x+w]

            # Compare the face region with known images
            match_results = [cv2.matchTemplate(face_roi, known_image, cv2.TM_CCOEFF_NORMED) for known_image in known_images_gray]

            if match_results:  # Check if match_results is not empty
                # Get the index of the best match
                best_match_index = np.argmax([np.max(result) for result in match_results])

                # # Check if the best match score exceeds the threshold and if the index is valid
                if best_match_index < len(person_names):
                #     # Display the name if a match is found
                    name = person_names[best_match_index]
                #     # Draw the name text on the resulting frame
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run()
