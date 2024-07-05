# **🙌 Hand Signs Classification using CNN Architecture/Transfer Learning 🤖**

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/2600/1*n0bmRzm1oSfagjnPdGbjMA.jpeg" alt="Leonardo AI">
</p>

This repository hosts my jupyter notebooks, where I've two separate techniques to train a ML model for classifying hand signs from 0 to 5 using the [[DL.AI] Hand Signs 05 Dataset](https://www.kaggle.com/datasets/shivamaggarwal513/dlai-hand-signs-05/data) from Kaggle 📊. First technique leverages *CNN architecture* and next-in-line uses *Transfer Learning*. You can visit these notebooks to refer each technique:

  - ***[CNN Architecture Model](https://github.com/PranayJagtap06/ML_Projects/blob/d9658b8b2b821030dfadc8b5d32cfe780ad4707a/Hand_Signs_Classification/hand_signs_multiclasss_classification.ipynb)***
  - ***[Transfer Learning Model](https://github.com/PranayJagtap06/ML_Projects/blob/d9658b8b2b821030dfadc8b5d32cfe780ad4707a/Hand_Signs_Classification/handsigns_transfer_learning.ipynb)***
  - ***[Open Transfer Learning Model Kaggle Notebook](https://www.kaggle.com/code/pranayjagtap06/handsigns-transfer-learning/notebook)***

The best model among these was deployed on Streamlit.

*[Visit Hand Signs Classifier Streamlit App](https://0-5handsignclassifier.streamlit.app/)*

*[Visit Hand Signs Classifier Streamlit App GitHub Repo](https://github.com/PranayJagtap06/0-5_Hand_Sign_Classifier)*

Here is a brief comparision of test set performance of produced models from these techniques:

  - CNN Architecture Model:
     - *Test set loss: 32.83%*
     - *Test set accuracy: 88.33%*
     - *Test set Categorical Accuracy: 1.0*
     - *Test set AUC-ROC score: 93.00%*
     - *Test set Classification Report:*
       
         | hand signs | precision | recall | f1-score | support |
         | ---- | ---- | ---- | ---- | ---- |
         | hand sign 0 | 1.00 | 1.00 | 1.00 | 11 |
         | hand sign 1 | 0.93 | 1.00 | 0.97 | 14 |
         | hand sign 2 | 0.71 | 0.71 | 0.71 | 7 |
         | hand sign 3 | 1.00 | 0.67 | 0.80 | 12 |
         | hand sign 4 | 0.78 | 1.00 | 0.88 | 7 |
         | hand sign 5 | 0.80 | 0.89 | 0.84 | 9 |


  - Transfer Learning Model:
    - *Test set loss: 28.21%*
    - *Test set accuracy: 93.33%*
    - *Test set Categorical Accuracy: 1.0*
    - *Test set AUC-ROC score: 96.00%*
    - *Test set Classification Report:*

        | hand signs | precision | recall | f1-scre | support |
        | ---- | ---- | ---- | ---- | ---- |
        | hand sign 0 | 1.00 | 1.00 | 1.00 | 10 |
        | hand sign 1 | 0.90 | 0.90 | 0.90 | 10 |
        | hand sign 2 | 0.90 | 0.90 | 0.90 | 10 |
        | hand sign 3 | 1.00 | 1.00 | 1.00 | 10 |
        | hand sign 4 | 0.83 | 1.00 | 0.91 | 10 |
        | hand sign 5 | 1.00 | 0.80 | 0.89 | 10 |

**Dataset Overview 📁**
---------------------

The [DL.AI] Hand Signs 05 Dataset comprises images of hand signs representing numbers 0 to 5 (inclusive) along with their corresponding labels, categorized into 6 distinct classes 📸.

**Size**: Dataset has 1200 labelled samples of size 15MB. Training set has 1080 samples and test set has 120 samples 🖼️.

**Image Resolution**: Images with height x width = 64 x 64 pixels and 3 channels (RGB) 📊.

Let's take a closer look at the file hierarchy…

    . train.h5 (HDF5 file)
    +-- train_set_x (HDF5 dataset)
    |   +-- 'numpy.ndarray' (1080, 64, 64, 3)
    +-- train_set_y (HDF5 dataset)
    |   +-- 'numpy.ndarray' (1080,)
    +-- list_classes (HDF5 dataset)
    |   +-- 'numpy.ndarray' (6,)

    . test.h5 (HDF5 file)
    +-- test_set_x (HDF5 dataset)
    |   +-- 'numpy.ndarray' (120, 64, 64, 3)
    +-- test_set_y (HDF5 dataset)
    |   +-- 'numpy.ndarray' (120,)
    +-- list_classes (HDF5 dataset)
    |   +-- 'numpy.ndarray' (6,)

## Classification with CNN Architecture

### **Preprocessing 🔧**
-----------------

Before training the model, I performed the following preprocessing steps:

* Normalized pixel values to the range [0, 1] ⚖️

### **CNN Model Architecture 🏗️**
-------------------------

The CNN model consists of the following layers:

Architecture:

| Hyperparameters/Layers | Values | Function |
| ----- | ----- | ----- |
| **Input Layer** | *tf.keras.layers.Input((width=64, height=64, color_channels=3), batch_size=32)* | Takes in target image and pre-processes for further layers |
| **Convolution Layers** | *tf.keras.layers.ConvXD* (X can be multiple values) | Learns most important features from target images |
| **Pooling Layers** | *tf.keras.layers.MaxPool2D* | Reduces the dimensionality of learned image features |
| **Flatten Layer** | *tf.keras.layers.Flatten* | Transforms 3D tensor to 1D vector |
| **Output Layer** | *tf.keras.layers.Dense* | Takes learned features and outputs them in shape of target labels |
| **Output Activation** | *'softmax' or tf.nn.Softmax* (for multiclass classification) | Adds non-linearity to output layer | 

Compilation:

| Attributes | Values | Function |
| ----- | ----- | ----- |
| **Optimizer** | *'adam' or tf.keras.optimizers.Adam* | Used to optimize the model's parameters during training |
| **Loss Function** | *tf.metrics.SparseCategoricalCrossentropy* |Computes the loss or penalty for each sample in the batch|
| **Metrics** | *'accuracy' or tf.metrics.Accuracy* | Used to evaluate the model's ability to correctly classify images |

### **Training and Evaluation 📊**
-------------------------
First a base model was trained using the Adam optimizer to find the ideal learning rate of the model. And the ideal learning rate was choosen as 0.002.

Then the model was trained with a learning rate of 0.002 and categorical cross-entropy loss 📈. The model was trained for 18 epochs with a batch size of 32 🕒.

The model achieved an accuracy of **88.33%** and categorical accuracy of **100.00%** on the testing set 🎉.

### **Preview Plots 📊**
-------------------------
*fig: Learning Rate vs Loss*

<p align="center">
  <img src="https://github.com/PranayJagtap06/ML_Projects/blob/main/Hand_Signs_Classification/assets/lossvslr.png" alt="Learning Rate vs Loss">
</p>

*fig: Losses vs Epoch*

<p align="center">
  <img src="https://github.com/PranayJagtap06/ML_Projects/blob/main/Hand_Signs_Classification/assets/lossesvsepoch.png" alt="Losses vs Epoch">
</p>

*fig: Accuracies vs Epoch*

<p align="center">
  <img src="https://github.com/PranayJagtap06/ML_Projects/blob/main/Hand_Signs_Classification/assets/accuraciesvsepoch.png" alt="Accuracies vs Epoch">
</p>

## Classification with Transfer Learning

In classification with transfer learning I'll use ***EfficientNetB0*** as base model of my image classification model.

### **Preprocessing 🔧**
-----------------

Before training the model, I performed the following preprocessing steps:

* Resizing images to size (224, 224) ⚖️

I skipped normalization as EfficientNetB0 already has an in-built normalization layer. Additionally, I'm also imcluding a data augmentaion layer in my transfer learning model. Below is a section on model architecture.

### **CNN Model Architecture 🏗️**
-------------------------

Transfer Learning Model has following layers:

Architecture:

| Hyperparameters/Layers | Name | Values |
| ----- | ----- | ----- |
| **InputLayer** | *input_layer* | *tf.keras.layers.Input(shape=(224, 224, 3), batch_size=32, name="input_layer")* |
| **Sequential** | *data_augmentation* | *tf.keras.models.Sequential([<data_augmentation_layers>], name="data_augmentation"* |
| **Functional** | *efficientnetb0* | *tf.keras.applications.efficientnet.EfficientNetB0(include_top=False)* |
| **GlobalAveragePooling2D** | *global_avg_pooling_layer* | *tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pooling_layer")* |
| **Dense** | *output_layer* | tf.keras.layers.Dense(6, activation="softmax", name="output_layer") |
| **Output Activation** | - | *'softmax' or tf.nn.Softmax* (for multiclass classification) |

Compilation:

| Attributes | Values |
| ----- | ----- |
| **Optimizer** | *'adam' or tf.keras.optimizers.Adam* |
| **Loss Function** | *tf.metrics.SparseCategoricalCrossentropy* |
| **Metrics** | *tf.keras.metrics.SparseCategoricalAccuracy()* |

### **Training and Evaluation 📊**
-------------------------

Here is the workflow I followed for training the model:

 - *Step1: Build and train a base model with Keras Functional API and data augmentation layer, with EfficientNetB0 as a base model, on only 10% of training data with Feature Extraction Transfer Learning.*
    - Validation set loss: 108.92%
    - Validation set sparse_categorical_accuracy: 75.00%
    - Validation set Categorical Accuracy: 1.0
    - Validation set AUC-ROC score: 85.00%
 
 - *Step2: Fine-Tune the base model on 10% of training data by making top 10 layers of EfficientNetB0 model as trainable and reducing the learning rate by -10%.*
    - Validation set loss: 67.49%
    - Validation set sparse_categorical_accuracy: 88.33%
    - Validation set Categorical Accuracy: 1.0
    - Validation set AUC-ROC score: 93.00%

 - *Step3: Re-train the fine-tuned base model on 100% of training data.*
    - Validation set loss: 22.40%
    - Validation set sparse_categorical_accuracy: 98.33%
    - Validation set Categorical Accuracy: 1.0
    - Validation set AUC-ROC score: 99.00%

### **Preview Plots 📊**
-------------------------

*fig: Final Model Loss & Accuracy Curve*

<p align="center">
  <img src="https://github.com/PranayJagtap06/ML_Projects/blob/main/Hand_Signs_Classification/assets/model_tl_val_la.png" alt="Loss & Accuracy Curve">
</p>

*fig: Final Model Precision Recall Curve*

<p align="center">
  <img src="https://github.com/PranayJagtap06/ML_Projects/blob/main/Hand_Signs_Classification/assets/model_tl_val_pr.png" alt="Precision Recall Curve">
</p>

*fig: Final Model ROC Curve*

<p align="center">
  <img src="https://github.com/PranayJagtap06/ML_Projects/blob/main/Hand_Signs_Classification/assets/model_tl_val_roc.png" alt="ROC Curve">
</p>

*fig: Final Model Test set Confusion Matrix*

<p align="center">
  <img src="https://github.com/PranayJagtap06/ML_Projects/blob/main/Hand_Signs_Classification/assets/model_tl_cm.png" alt="Confusion Matrix">
</p>

**Colab/Kaggle Notebooks 📝**
-----------------

The Colab/Kaggle notebooks in this repository provides a step-by-step guide to reproducing the results 📚. You can run the notebook in Google Colab/Kaggle to access GPU and explore the code and experiment with different hyperparameters 🔍.

**Getting Started 🚀**
-------------------

To get started, simply clone this repository and open the notebooks in Google Colab/Kaggle 💻. Make sure to install the required libraries and dependencies before running the notebooks 📦.

**Contributing 🤝**
--------------

If you'd like to contribute to this project, please feel free to open an issue or submit a pull request 📝. I'm always looking for ways to improve the model and explore new ideas 💡.

**License 📝**
---------

This repository is licensed under the MIT License 📜.

**Acknowledgments 🙏**
----------------

I'd like to thank Kaggle for providing the [DL.AI] Hand Signs 05 Dataset and Google Colab for providing a platform to develop and share this project 🙏.

**Happy Learning! 🤓**
-------------------

I hope you find this repository helpful in your machine learning journey! If you have any questions or need help, feel free to reach out 🤝.
