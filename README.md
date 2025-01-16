# Helmet and Number Plate Detection using YOLOv8

## Project Overview

The **"Helmet and Number Plate Detection using YOLOv8"** project is designed to automatically detect helmet usage and number plates in videos or images using the YOLOv8 (You Only Look Once) deep learning model. YOLOv8 is a powerful, state-of-the-art object detection model capable of detecting multiple objects in real-time. This project leverages YOLOv8’s capabilities to detect riders with or without helmets and capture the number plate information when necessary.

The system can be used in surveillance applications or vehicle monitoring to detect whether a rider is wearing a helmet and identify vehicles by their number plates. If a rider is not wearing a helmet, the number plate is detected and saved in a text file for further analysis.

---

## Features

- **Helmet Detection**: Detects riders wearing helmets or not using YOLOv8.
- **Number Plate Detection**: If a rider is not wearing a helmet, the number plate of the vehicle is detected.
- **Real-Time Detection**: Processes both images and videos for real-time helmet and number plate detection.
- **Output**: 
  - Bounding boxes around the detected objects (helmet and number plate).
  - Saves the detected number plates in a `.txt` file for future reference.
- **Customizable**: You can use your own dataset for helmet and number plate detection.
  
---

## Technologies Used

- **YOLOv8**: A deep learning model used for object detection.
- **OpenCV**: A computer vision library for image and video processing.
- **Python**: Programming language used for the implementation.
- **PyTorch**: Framework for training and using YOLOv8 models.

---

## Installation

### Prerequisites

Make sure you have the following installed on your machine:

- Python 3.x
- CUDA (for GPU acceleration, optional but recommended)

### Install Required Libraries

You can install the required dependencies using the following commands:

```bash
pip install ultralytics
pip install opencv-python
pip install matplotlib
```

---

## Dataset

The dataset used in this project contains images of riders with and without helmets, along with vehicle number plates. The dataset is structured as follows:

- **Images**: Folder containing training images.
- **Labels**: Folder containing corresponding labels in YOLO format.
  
The project uses **YOLOv8** for training the model. You can modify the dataset and labels according to your requirements.

---

## Usage

### Training the YOLOv8 Model

To train the model with your own dataset, use the following command:

```bash
yolo task=detect mode=train model=yolov8s.pt data=/path/to/your/dataset.yaml epochs=10 project=/path/to/save/results name=results
```

Where:
- `data`: Path to your dataset YAML file that contains the paths to the training and validation data.
- `epochs`: Number of epochs to train the model (adjust based on your dataset).
- `project`: The directory where training results will be saved.
- `name`: The name for saving the trained weights and results.

### Detecting Objects in Images or Videos

After training the model, you can use it to detect helmets and number plates in images or videos.

To process an image:

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('/path/to/trained/model/best.pt')

# Perform detection on the image
result = model('/path/to/your/image.jpg')

# Save the result
result[0].save("/path/to/save/output/image_result.jpg")
```

To process a video and save the results:

```python
import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('/path/to/trained/model/best.pt')

# Open video
cap = cv2.VideoCapture('/path/to/your/video.mp4')

# Open file to save number plates
output_txt_path = '/path/to/save/number_plates.txt'
with open(output_txt_path, 'w') as txt_file:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection
        results = model(frame)

        # Process results
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = box.conf
                xyxy = box.xyxy.cpu().numpy().astype(int)
                class_name = model.names[cls_id]

                if class_name == "without helmet":
                    # Draw bounding box for "without helmet"
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
                    cv2.putText(frame, f"Without Helmet {conf:.2f}", (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                elif class_name == "number plate":
                    # Save detected number plate coordinates to text file
                    txt_file.write(f"Frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}: Number Plate at {xyxy.tolist()}\n")

                    # Draw bounding box for "number plate"
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
                    cv2.putText(frame, f"Number Plate {conf:.2f}", (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Display the result
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
```

### Output

- For images, the bounding boxes for helmets and number plates are drawn, and the modified image is saved.
- For videos, bounding boxes are drawn in real-time, and any detected number plates are saved in a `.txt` file.

---

## Results

Once the model has been trained and applied to your dataset, it should be able to detect helmets and number plates accurately. You can fine-tune the model by adjusting the number of epochs, batch size, or other training parameters based on your dataset.

---

## Contributing

If you'd like to contribute to this project, feel free to fork the repository, create a pull request with improvements, or report issues you encounter.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

