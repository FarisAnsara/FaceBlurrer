# Face Blurring Algorithm

## Overview
This project implements a face blurring algorithm that detects and blurs faces in any given colored image using a pre-trained YOLOv8 model. If no faces are detected, the image remains unchanged. The algorithm ensures people's privacy by blurring their faces while preserving the rest of the image.

---

## Dataset
The dataset used for this project can be found on [Kaggle](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset).  
- **Dataset location:** Place the downloaded ZIP file in the `FacesDataSet` directory.  
- **Extraction:** Use the provided Python script to extract the dataset contents.

---

## Requirements
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

**Dependencies:**
- Python == 3.11.3
- `numpy`
- `pandas`
- `matplotlib`
- `Pillow`
- `ultralytics`
- `opencv-python`

---

## How to Run
### 1. **Setting Up the Environment**
   - Ensure you have Python 3.11.3 installed.
   - Install the dependencies using:
     ```bash
     pip install -r requirements.txt
     ```

### 2. **Running the Model**
   - To run the face detection and blurring script:
     ```bash
     python Main.py
     ```
   - Replace the `input_image_path` and `output_image_path` variables in `Main.py` with the desired input and output file paths.

---

## Documentation
### **Model Training**
- **Pre-trained Model Used:** YOLOv8n  
- **Training Dataset:** Images with face annotations were used to fine-tune the model.  
- **Approach:** Training details and results are documented in the provided Jupyter Notebook `FaceDetectionModelTraining.ipynb`. For a detailed analysis of the training process, please refer to this notebook.
- **Pre-trained Model Used:** YOLOv8n  
- **Training Dataset:** Images with face annotations were used to fine-tune the model.  
- **Approach:** Training details and results are documented in the provided Jupyter Notebook.  

### **Testing**
- The model is tested on a separate validation dataset to evaluate its performance using metrics such as Precision, Recall, mAP50, and mAP50-95. For a comprehensive analysis of the testing results, please refer to the Jupyter Notebook `FaceDetectionModelTraining.ipynb`.
- The model is tested on a separate validation dataset to evaluate its performance using metrics such as Precision, Recall, mAP50, and mAP50-95.

### **Face Blurring**
- A detected face is blurred using Gaussian blurring with customizable blur strength.
- If no face is detected, a message is displayed, and the image remains unaltered.

---

## Results
- **Training Results:** Documented in the Jupyter Notebook.
- **Testing Results:** Confusion matrices, precision-recall curves, and mAP scores are included in the `detect_faces/test_evaluation` directory.

---

## Future Enhancements
1. Optimize the model for real-time video processing.
2. Improve detection accuracy for occluded and low-light faces.
3. Extend the functionality to detect and anonymize other sensitive objects.

---

## Notes
- This project is designed to work seamlessly on both Windows and Linux platforms.
- For any issues, ensure the paths to the dataset and pre-trained model are correctly specified.