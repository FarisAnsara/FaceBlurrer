# Face Blurring Algorithm

## Overview
This project implements a face blurring algorithm that detects and blurs faces in any given colored image using a pre-trained YOLOv8 model. If no faces are detected, the image remains unchanged. The algorithm ensures people's privacy by blurring their faces while preserving the rest of the image.

---

## Dataset
The dataset used for this project can be found on [Kaggle](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset).  
- **Dataset location:** Place the downloaded ZIP file in the `FacesDataSet` directory.
- **Extraction:** Use the provided Python script to extract the dataset contents.
```python
import zipfile

zip_path = os.path.join('FaceMaskDataSet', 'archive.zip')
dataset_path = os.path.join('FaceMaskDataSet')

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dataset_path)
```

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

## Cloning the Repository
To clone this repository onto your local machine:
1. Use the following command:
```bash
git clone https://github.com/FarisAnsara/FaceBlurrer.git
```
2. Navigate to the project directory:
```bash
cd FaceBlurrer
```

---

## How to Run
### 1. **Setting Up the Environment**
   - Ensure you have Python 3.11.3 installed.
   - Install the dependencies using:
     ```bash
     pip install -r requirements.txt
     ```

### 2. **Running the Model**
   - To run the face detection and blurring script, use the following command:
     ```bash
     python Main.py --input_image_path <path_to_input_image> --output_image_path <path_to_output_image>
     ```
   - Replace `<path_to_input_image>` and `<path_to_output_image>` with the actual file paths. 

   - If the `--output_image_path` is provided only as a directory, the script will automatically save the output file in that directory, appending `_blurred` to the original input file name.

   - Alternatively, you can run the script without any arguments to rely on the default input and output paths:
     ```bash
     python Main.py
     ```
     In this case, the script will:
     - Use the default input image path specified in the `Main.py` file.
     - Save the blurred image automatically in the `FacesDataSet/images/blurred_faces` directory with `_blurred` appended to the original file name.

   - If using the default paths, ensure you have updated the `input_image_path` variable directly in the `Main.py` file as needed.

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
2. Additional future enhancements are detailed in the `FaceDetectionModelTraining.ipynb` Jupyter Notebook.

---

## Notes
- This project is designed to work seamlessly on both Windows and Linux platforms.
- For any issues, ensure the paths to the dataset and pre-trained model are correctly specified.

