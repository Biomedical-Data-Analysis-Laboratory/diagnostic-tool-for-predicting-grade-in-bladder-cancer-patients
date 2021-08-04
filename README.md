# Automatic diagnostic tool for predicting cancer grade in bladder cancer patients using deep learning

This is the source code described in the paper "Automatic diagnostic tool for predicting cancer grade in bladder cancer patients using deep learning" by Rune Wetteland, Vebjørn Kvikstad, Trygve Eftestøl, Erlend Tøssebro, Melinda Lillesand, Emiel A.M. Janssen, and Kjersti Engan.

### Abstract
The most common type of bladder cancer is urothelial carcinoma, which is among the cancer types with the highest recurrence rate and lifetime treatment cost per patient. Diagnosed patients are stratified into risk groups, mainly based on grade and stage. However, it is well known that correct grading of bladder cancer suffers from intra- and interobserver variability and inconsistent reproducibility between pathologists, potentially leading to under- or overtreatment of the patients. The economic burden, unnecessary patient suffering, and additional load on the health care system illustrate the importance of developing new tools to aid pathologists. We propose a pipeline, called TRI_grade, that will identify diagnostic relevant regions in the whole-slide image (WSI) and collectively predict the grade of the current WSI. The system consists of two main models, trained on weak slide-level grade labels. First, a WSI is segmented into the different tissue classes (urothelium, stroma, muscle, blood, damaged tissue, and background). Next, tiles are extracted from the diagnostic relevant urothelium tissue from three magnification levels (25x, 100x, and 400x) and processed sequentially by a convolutional neural network (CNN) based model. Ten models were trained for the slide-level grading experiment, where the best model achieved an F1-score of 0.90 on a test set consisting of 50 WSIs. The best model was further evaluated on a smaller segmentation test set, consisting of 14 WSIs where low- and high-grade regions were annotated by a pathologist. The TRI_grade pipeline achieved an average F1-score of 0.91 for both the low-grade and high-grade classes.

![alt text](images/segmentation_vs_groundtruth.png?raw=true)

### Requirements

The code was built using the following Python packages:

python==3.6.7  
numpy==1.18.5  
opencv-python==4.4.0.42  
scikit-image==0.17.2  
scipy==1.4.1  
pyvips==2.1.12
tensorflow==1.13.1  
keras==2.2.4  
matplotlib==3.1.0  
PIL==6.0.0  

### Link to paper
TBA

### How to cite our work
The code is released free of charge as open-source software under the GPL-3.0 License. Please cite our paper if you use it in your research.
```
@article{wetteland2021diagnostic,
  title={Automatic diagnostic tool for predicting cancer grade in bladder cancer patients using deep learning},
  author={Wetteland, Rune and Kvikstad, Vebjørn and Eftestøl, Trygve and Tøssebro, Erlend and Lillesand, Melinda and Janssen, Emiel A.M. and Engan, Kjersti},
  journal={To be published in IEEE Open Journal of Engineering in Medicine and Biology},
  year={2021},
  publisher={IEEE}
}
```
