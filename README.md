# Breast Cancer Classification Using Neural Network

This project is focused on classifying breast cancer as either malignant or benign using a neural network (NN) model. It uses the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, which contains various features computed from digitized images of breast masses. 

## Dataset

- **Source**: [UCI Machine Learning Repository](https://goo.gl/U2Uwz2)
- **Dataset Size**: 569 samples
- **Features**: 30 numerical features computed from images of fine needle aspirates (FNAs) of breast masses.
- **Classes**: 2 classes (Malignant - 212, Benign - 357)

## Project Structure

- **Data Preprocessing**: 
  - The data is cleaned and normalized to prepare it for the NN model.
  - Missing values are handled, and feature scaling is applied.
  
- **Model Architecture**: 
  - A simple feed-forward neural network is built using TensorFlow/Keras.
  - The model uses layers with activation functions like ReLU and softmax for final classification.

- **Training and Evaluation**: 
  - The model is trained on the dataset, and various metrics such as accuracy, precision, recall, and F1-score are used for evaluation.
  - Cross-validation and test sets are employed for performance benchmarking.

## Usage

To run the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/abbasmithaiwala/breast-cancer-classification-using-nn
   cd breast-cancer-classification-using-nn
   ```
2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook:

    ```bash
    jupyter notebook Breast\ Cancer\ Classification\ using\ NN.ipynb
    ```

## Requirements

* Python 3.x
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib

## Results

* The NN model achieved high classification accuracy on the test set, providing valuable insights into the features that contribute most to breast cancer prediction.

## Acknowledgments

* This project utilizes the dataset made available by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

