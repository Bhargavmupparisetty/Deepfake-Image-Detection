# DeepFake Detector — Django Web App (Image & Video Classification)

This project is a deep learning–powered web application built with **Django** to detect deepfake media. It supports both **image** and **video** uploads and classifies them as **Real** or **Manipulated** using a fine-tuned version of the [MesoNet-4](https://github.com/DariusAf/MesoNet) architecture.


##  Features

-  Upload **images** or **videos**
-  Finetuned **MesoNet-4** model on a large real/fake dataset
-  Live classification with **confidence score bar**
-  Visualizations: Accuracy, Confusion Matrix, Architecture
-  Simple Django web interface for usability



##  Screenshots

### UI – Upload & Prediction Interface

![App UI](images/test.png)


### Model Architecture

![Architecture](images/architecture.png)

### Fine-tuning Accuracy & Loss Graphs

![Training Metrics](images/finetune_metrics.png)

### Confusion Matrix on Validation Set

![Confusion Matrix](images/confusion_matrix.png)



##  Model Details

- Architecture: MesoNet-4 ([Original GitHub Repo](https://github.com/DariusAf/MesoNet))
-  Weights used: `mesonet4_DF.h5` (pretrained) + **custom finetuning**
-  Task: Binary classification — `Real` vs `Fake`
-  Dataset: Combined real/fake dataset from FaceForensics++, Celeb-DF, and other sources
-  Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix


##  Setup Instructions

### 1. Clone the repo

```bash
      git clone https://github.com/BhargavMupparisetty/Deepfake-Image-Detection.git
      cd deepfake-detector
```


### 2. Create and activate virtual environment

```bash
      python -m venv venv
      venv\Scripts\activate   # On Windows
      source venv/bin/activate  # On Linux/macOS
```

### 3. Run the Django app

```bash
   python manage.py runserver
```


Then visit:

```bash
 http://127.0.0.1:8000/
````


## Acknowledgements

 MesoNet-4: This project builds upon DariusAf/MesoNet, which provided the foundational model architecture.

 ### Datasets:
 
- FaceForensics++
- Celeb-DF
- DeepFakeDetection Challenge Dataset

 ### License
 
This project is licensed under the MIT License.

### Future Plans

- Add Grad-CAM or saliency map for explainability
- Add support for real-time webcam feed
- Host the web app in cloud

### Author
Developed by Bhargav Mupparisetty
