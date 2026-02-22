# SEP: Computer Vision & Deep Learning
Deep learning–based Facial Emotion Recognition featuring Grad-CAM explainability and real-time interactive inference.


## Overview
This project is the result of the "Software Development Practical" held at the Ludwig Maximilian University of Munich (LMU) during the Winter Semester 2025/26.

Within the topic *Computer Vision & Deep Learning*, students were tasked with training and evaluating a Convolutional Neural Network for Facial Emotion Recognition. We developed a fully end-to-end system in which all preprocessing, training, evaluation, and XAI-related pipelines were implemented form scratch.

The practical course was supervised by Johannes Schusterbauer and Ming Gui, PhD students at LMU.


## Team Members
- Alen Alija
- Kilian Killer
- Leon Maibauer
- Richard Tscharke


## Installation

1. Clone the repository.
```bash
git clone https://github.com/RichardTscharke/SEP-Alija-Killer-Maibauer-Tscharke.git
cd SEP-Alija-Killer-Maibauer-Tscharke
```
2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Install ONNX Runtime (required for InsightFace):

**Linux with CUDA** GPU support:
```bash
pip install onnxruntime-gpu
```
**macOS/CPU** systems:
```bash
pip install onnxruntime
```


## Quick Start (Pretrained Model)

If you do **not** want to train the model from scratch but instead use our pretrained model for inference and evaluation, run `main.py` by calling:

```bash
python main.py
```

The script will guide you through:

- CSV file generation for custom datasets
- Grad-CAM–based explainable AI for image and/or video input
- An interactive live demo using your webcam

Alternatively, run

```bash
python -m src.explain_image
```

or

```bash
python -m src.explain_video
```

with a path argument. You can also run the commands without any arguments to view the Grad-CAMs generated for our included default image or video (`xai_samples/`).


## Training from Scratch

If you want to train your own model with custom dataset distributions and training configurations, start by downloading the following datasets:

### RAF-DB (Real-world Affective Faces Database)

1. Visit the [official RAF-DB website](http://www.whdeng.cn/RAF/model1.html#dataset).
2. Go to the "How to get the Password" section and follow the instructions.
3. In the email you receive, open one of the Basic emotion links.
4. Download and unzip the `Image` and `EmoLabel` folders (including all contents).
5. Place them both into a new folder and **rename** it `RAF`.

### ExpW (Expression in the Wild)

1. Visit the [official ExpW website](https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html).
2. Go to the "Downloads" section and open the ExpW link.
3. Download the `data` folder (make sure it contains an `image` folder with the 8 archive parts and a `label` folder containing `label.lst`).
4. Open a terminal inside the `image` folder and run:
   
```bash
7z x origin.7z.001
```

*(Make sure `p7zip` is installed.)*

5. **Rename** the `data` folder to `ExpW`.

### KDEF (Karolinska Directed Emotional Faces)

1. Visit the official [KDEF website](https://kdef.se/).
2. Go to "Download" -> "Download the original KDEF and AKDEF" and fill out the form.
3. Press the Download button.
4. Download and unzip the original KDEF & AKDEF material.
5. Within the `KDEF_and_AKDEF` folder extract the `KDEF` folder.

Finally, place all three folders (`RAF`, `ExpW`, `KDEF`) into the `data` folder within the project root.

Alternatively, you can manually fill the 6 emotion folders per `train`, `test`, and `validate` directory within the `data` directory with your own images. 

Note: The folder structure is included using .gitkeep files for demonstration purposes. During normal preprocessing, these directories are created automatically.


## Custom Configuration

After successfully downloading and placing the three datasets, you can configure your own data splits and training metrics for custom setups:
1. Adjust and call the `src/preprocess.py` script.
2. Adjust and call the `src/train.py` script.
3. Evaluate your own setup by reviewing performance-related graphics automatically created during evaluation within the `figures` folder.
4. Change the model path within `src/explain_image.py`, `src/explain_video.py`, and `src/run_demo.py` to your new path which can be found in the `model_paths` folder.
5. Run `main.py` or Call any of these scripts for Grad-CAM-based explanations for images, videos or webcam input.

Please note that all important information is documented within the respective files.


---

Ludwig Maximilian University of Munich (LMU)  
Software Development Practical — Winter Semester 2025/26
