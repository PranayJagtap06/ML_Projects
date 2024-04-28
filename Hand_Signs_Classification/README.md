# **Hand Signs Classification using CNN & TensorFlow 🤖**

**Welcome to Hand Signs Classification! 🙌**
------------------------------------------

This repository hosts my Colab notebook, where I've trained a Convolutional Neural Network (CNN) model using TensorFlow to classify hand signs from 0 to 5 using the [[DL.AI] Hand Signs 05 Dataset](https://www.kaggle.com/datasets/shivamaggarwal513/dlai-hand-signs-05/data) from Kaggle 📊.

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

**Preprocessing 🔧**
-----------------

Before training the model, I performed the following preprocessing steps:

* Normalized pixel values to the range [0, 1] ⚖️

**CNN Model Architecture 🏗️**
-------------------------

The CNN model consists of the following layers:

Architecture:

| Hyperparameters/Layers | Values | Function |
| ----- | ----- | ----- |
| **Input** | *Image or Video* | Target image to learn patterns |
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

**Training and Evaluation 📊**
-------------------------
First a base model was trained using the Adam optimizer to find the ideal learning rate of the model. And the ideal learning rate was choosen as 0.002.

Then the model was trained with a learning rate of 0.002 and categorical cross-entropy loss 📈. The model was trained for 18 epochs with a batch size of 32 🕒.

The model achieved an accuracy of **88.33%** and categorical accuracy of **100.00%** on the testing set 🎉.

**Colab Notebook 📝**
-----------------

The Colab notebook in this repository provides a step-by-step guide to reproducing the results 📚. You can run the notebook in Google Colab to explore the code and experiment with different hyperparameters 🔍.

**Getting Started 🚀**
-------------------

To get started, simply clone this repository and open the Colab notebook in Google Colab 💻. Make sure to install the required libraries and dependencies before running the notebook 📦.

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
