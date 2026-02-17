# PCB Defect Detection 

A machine learning project that detects and identifies defects in printed circuit boards (PCBs) with a YOLO convolutional neural network.

## Project Overview

This model indicates whether there are defects in images of PCBs or not, and which of the 6 defect classes (missing hole, mouse bite, open circuit, short circuit, spur, and spurious copper) the images belong to.

---

## Architecture and Data Pipeline

Raw dataset in VOC XML format
    ↓
convert.py creates text files containing information on the bounding boxes and classification
    ↓
split.py performs a basic 80/20 training/validation data split
    ↓
train.py searches for the file paths found in the yaml file and trains the YOLO model with it
    ↓
test_model.py tests and validates training results
    ↓
api.py deploys model via a FastAPI

---

## Model Performance

mean average precision: ranges between 94.5 and 95% post training
inference speed: (1ms dependent on hardware specification)
total time per 100 epochs: approximately 38 minutes (Ryzen 9 7950X, NVIDIA 5070ti)

---

## Key Learnings and Results

- XML to text conversion
- yaml file creation
- tunung training parameters and analyzing training results (loss, mAP by class, etc)
- simple machine learning data flow
- FastAPI deployment

## Potential Improvements

- possible MLFlow integration
- multi-batch implementation
- custom confidence thresholding
- database logging (PostgreSQL)
- int8 quantization

---

## Acknowledgements

Dataset: [Link](https://www.kaggle.com/code/sunnyconsultant/pcb-defects-detected-by-yolo-v5)

Built as a practical learning experience to demonstrate: 
- Training a multi-layered convolutional neural network with obtained data
- Advanced file format navigation, parsing, and conversion
- Familiarity with GPU usage in machine learning / AI related professions
- Complex debug capabilities

---

**Last Updated:** January 2026