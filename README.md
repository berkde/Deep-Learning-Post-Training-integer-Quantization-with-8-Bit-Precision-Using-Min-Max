#   Deep Learning Research: Analysis of Post-Training Integer Quantization with 8-Bit Precision Using Min-Max Scheme on MNIST for Digit Detection

This project demonstrates how to train a TensorFlow model to recognize handwritten digits from the MNIST dataset and then optimize it for deployment using TensorFlow Lite. The optimization techniques used are dynamic range quantization and full integer quantization.
##   Project Overview
The script performs the following:
1.  **Loads and Preprocesses the MNIST Dataset:** The MNIST dataset is loaded using `tf.keras.datasets.mnist.load_data()` and the images are normalized.
2.  **Builds and Trains a Convolutional Neural Network (CNN):** A simple CNN is created using `tf.keras.models.Sequential` and trained on the MNIST training data.
3.  **Evaluates the Baseline Model:** The trained model's accuracy is evaluated on the test dataset.
4.  **Saves the Baseline Model:** The model is saved in the H5 format.
5.  **Converts to TensorFlow Lite:** The saved Keras model is converted to the TensorFlow Lite format.
6.  **Applies Dynamic Range Quantization:** The TensorFlow Lite model is quantized using dynamic range quantization to reduce the model size.
7.  **Applies Full Integer Quantization:** The TensorFlow Lite model is further quantized to full integer quantization for maximum optimization. This involves providing a representative dataset.
8.  **Saves the Quantized Models:** Both the dynamically quantized and the fully integer quantized models are saved as `.tflite` files.
9.  **Evaluates TensorFlow Lite Model Accuracy:** The accuracy of the quantized TensorFlow Lite models is evaluated.
10. **Compares Model Sizes:** The file sizes of the original Keras model and the quantized TensorFlow Lite models are compared.
11. **Measures Inference Time:** The inference time for the baseline Keras model and the TensorFlow Lite models are measured and compared.
##   Prerequisites
* Python 3.x
* TensorFlow 2.x
##   Installation
1.  Install TensorFlow:
 ```bash
 pip install numpy pandas tensorflow notebook jupyterlab
 pip install mistune
 pip install --upgrade nbconvert
   ```
##   Usage
1.  Run the Python script to train the model, quantize it, and evaluate the results:
 ```bash
 jupyter nbconvert --execute PTQ8.ipynb --to notebook
   ```
##   Results

* 

    **Model Performance and Size Comparison**

    | Metric                                                              | Baseline Keras Model | Dynamically Quantized TFLite Model | Fully Integer Quantized TFLite Model |
    | :------------------------------------------------------------------ |:---------------------| :--------------------------------- | :----------------------------------- |
    | Accuracy                                                              | 0.9834               | 0.9834               | 0.9831                   |
    | File Size (approximate for Baseline)                                  | 661.91 KB            | 214.85 KB           | 57.20 KB             |
    | Inference Time (ms/sample)                                            | 14.4569              | 0.0233                  | 0.0198               |



##   Model Files
The script generates the following model files:
* `mnist_baseline.h5`: The saved Keras model.
* `mnist_quantized_dynamic_range.tflite`: The dynamically quantized TensorFlow Lite model.
* `mnist_quantized_full_int.tflite`: The fully integer quantized TensorFlow Lite model.
##   Optimization Details
* **Dynamic Range Quantization:** This technique reduces the size of the model by quantizing the weights from floating-point to 8-bit integers.
* **Full Integer Quantization:** This technique quantizes both the weights and the activations to 8-bit integers, potentially providing further size reduction and inference speedup.
##   Disclaimer
* The baseline model size is approximated from the saved H5 file.
* Warnings related to TensorFlow Lite conversion and SavedModel may appear during execution.