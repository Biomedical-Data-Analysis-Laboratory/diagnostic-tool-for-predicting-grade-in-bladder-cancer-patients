# Automatic diagnostic tool for predicting cancer grade in bladder cancer patients using deep learning

This is the source code described in the paper "Automatic diagnostic tool for predicting cancer grade in bladder cancer patients using deep learning" by Rune Wetteland, Vebjørn Kvikstad, Trygve Eftestøl, Erlend Tøssebro, Melinda Lillesand, Emiel A.M. Janssen, and Kjersti Engan.

### 1 -Abstract
The most common type of bladder cancer is urothelial carcinoma, which is among the cancer types with the highest recurrence rate and lifetime treatment cost per patient. Diagnosed patients are stratified into risk groups, mainly based on grade and stage. However, it is well known that correct grading of bladder cancer suffers from intra- and interobserver variability and inconsistent reproducibility between pathologists, potentially leading to under- or overtreatment of the patients. The economic burden, unnecessary patient suffering, and additional load on the health care system illustrate the importance of developing new tools to aid pathologists. We propose a pipeline, called TRI_grade, that will identify diagnostic relevant regions in the whole-slide image (WSI) and collectively predict the grade of the current WSI. The system consists of two main models, trained on weak slide-level grade labels. First, a WSI is segmented into the different tissue classes (urothelium, stroma, muscle, blood, damaged tissue, and background). Next, tiles are extracted from the diagnostic relevant urothelium tissue from three magnification levels (25x, 100x, and 400x) and processed sequentially by a convolutional neural network (CNN) based model. Ten models were trained for the slide-level grading experiment, where the best model achieved an F1-score of 0.90 on a test set consisting of 50 WSIs. The best model was further evaluated on a smaller segmentation test set, consisting of 14 WSIs where low- and high-grade regions were annotated by a pathologist. The TRI_grade pipeline achieved an average F1-score of 0.91 for both the low-grade and high-grade classes.

![alt text](images/Proposed_system.png?raw=true)

![alt text](images/segmentation_vs_groundtruth.png?raw=true)

### 2 - How to use

#### 2.1 – Diagnostic model

The main file contains three different 'modes' to choose from. To enable a mode, set the top variable under that mode to True. To run the main file, add the argument "True" or "False". "True" means you are starting a new model, and a new folder will be created. "False" will resume an existing model. If you have multiple models, use the variable CONTINUE_FROM_MODEL to specify which model to start from. By default, CONTINUE_FROM_MODEL is set to "last", which will resume the most recent model.

The three modes are:
*	Train a diagnostic model using training and validation dataset
*	Use the trained model to find the optimal slide-level decision threshold
*	Test a diagnostic model on the test dataset using the decision threshold

The program can train diagnostic models for five different labels (WHO73, WHO04, Stage, Recurrence, and Progression).

The code contains a large list of options. From the main file, it is possible to choose which base model to use (VGG16/VGG19), freeze/unfreeze base model, dropout rate, number of neurons in the classifier layers, how many scales to use, and which scales to use. There is a debug mode where only a small dataset is used. There are some options for data augmentation. If the training is stopped due to, e.g., a power outage, it is possible to continue training from the last epoch; this also workes if training multiple models. It is possible to enter a list of hyperparameters, e.g. learning rate = [0.1, 0.01] dropout_rate = [0.1, 0.4], and the program will train all possible combinations of the hyperparameters. The program will save learning curves, model plots, confusion matrices, classification reports, and logs.

Place the dataset in a folder called “WSI_files/”, where each SCN image has a separate folder (e.g. “WSI_files/H1234/H1234.scn”.

To generate a dataset to train the diagnostic model, first use the tissue segmentation program from here:

https://github.com/Biomedical-Data-Analysis-Laboratory/multiscale-tissue-segmentation-for-urothelial-carcinoma

Then, use the following program to extract the urothelium tiles:

https://github.com/Biomedical-Data-Analysis-Laboratory/extracting-tiles-in-multilevel-gigapixel-images

Place the generated pickle coordinate files in the diagnostic model folder, and the model will use them for training a new model.

#### 2.2 Requirements

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

### 3 - Link to paper
https://ieeexplore.ieee.org/document/9513308

### 4 - How to cite our work
The code is released free of charge as open-source software under the GPL-3.0 License. Please cite our paper if you use it in your research.
```
@article{wetteland2021diagnostic,
  title={Automatic diagnostic tool for predicting cancer grade in bladder cancer patients using deep learning},
  author={Wetteland, Rune and Kvikstad, Vebjørn and Eftestøl, Trygve and Tøssebro, Erlend and Lillesand, Melinda and Janssen, Emiel A.M. and Engan, Kjersti},
  journal={To be published in IEEE Open Journal of Engineering in Medicine and Biology},
  year={2021},
  publisher={IEEE},
  url={doi.org/10.1109/ACCESS.2021.3104724}
}
```
