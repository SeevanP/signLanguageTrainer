import os
import cv2

#data_dir = 
number_of_classes = 18
dataset_size = 100

cap = cv2.VideoCapture(0)  # Change camera index if needed

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

for j in range(number_of_classes):
    # Create folder if it doesn't exist
    class_dir = os.path.join(data_dir, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Wait for user to be ready
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(100) == ord('q'):  # Increased wait time for better interaction
            break

    # Capture images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        cv2.imshow('frame', frame)
        cv2.waitKey(100)  # Increased wait time for better control
        
        # Save the frame as an image
        img_path = os.path.join(class_dir, '{}.jpg'.format(counter))
        cv2.imwrite(img_path, frame)
        print(f"Saved image {img_path}")

        counter += 1

cap.release()
cv2.destroyAllWindows()
