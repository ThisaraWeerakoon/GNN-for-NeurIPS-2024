
<div align="center">    
 
# Predict small molecule-protein interactions using BELKA   

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
	![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)


<!--  
Conference   
-->   
</div>


![images (11)](https://github.com/user-attachments/assets/cd1090d6-261f-4765-929a-7a5273f749fe)


## Description   
This project aims to develop machine learning model to predict the binding affinity of small molecules to specific protein targets.

- **Motivation:** This is a critical step in drug development for the pharmaceutical industry that would pave the way for more accurate drug discovery
- **Why:** There are likely effective treatments for human ailments hiding in that chemical space, and better methods to find such treatments are desirable to us all.Searching chemical space by inference using this type of computational models rather than running laboratory experiments contribute to advances in small molecule chemistry used to accelerate drug discovery.

- **Problem Solved:** The number of chemicals in druglike space has been estimated to be 10^60, a space far too big to physically search.ML approaches suggest it might be possible to search chemical space by inference using well-trained computational models rather than running laboratory experiments.
- **What I Learned:**
  - **1:** Fundamentals of Graph Neural Networks
  - **2:** Working with chemical data including chemical specifications like <b>Simplified Molecular-Input Line-Entry System (SMILES)</b>
  - **3:** Working with analytical database like <b>DuckDB</b>

## Background
Small molecule drugs are chemicals that interact with cellular protein machinery and affect the functions of this machinery in some way. Often, drugs are meant to inhibit the activity of single protein targets, and those targets are thought to be involved in a disease process. A classic approach to identify such candidate molecules is to physically make them, one by one, and then expose them to the protein target of interest and test if the two interact. This can be a fairly laborious and time-intensive process.

## Dataset
To evaluate potential search methods in small molecule chemistry, <a href="https://www.leash.bio/">Leash Biosciences</a> physically tested some 133M small molecules for their ability to interact with one of three protein targets using DNA-encoded chemical library (DEL) technology. This dataset, the <b>Big Encoded Library for Chemical Assessment (BELKA)</b>, provides an excellent opportunity to develop predictive models that may advance drug discovery.



 ## Methodology

1. **Data Visualization:**
   - We used the UTKFace dataset, comprising over 20,000 facial images with annotations of age, gender, and ethnicity.
   - Images were visualized to understand the dataset distribution and the embedded labels.

2. **Data Preprocessing:**
   - Images were resized to 224x224 pixels to match the VGG16 model requirements.
   - Normalization was performed to scale pixel values between 0 and 1.
   - Age labels were extracted and categorized into five age groups: 0–24, 25–49, 50–74, 75–99, and 100–124.

3. **Transfer Learning with VGG16:**
   - The VGG16 model, pre-trained on ImageNet, was used as the base model.
   - The model's layers were frozen, and additional dense layers with dropout and L2 regularization were added.
   - The final output layer was designed to classify images into the five age groups using softmax activation.

4. **Model Training:**
   - The model was compiled with categorical cross-entropy loss and the Adam optimizer.
   - Early stopping and model checkpoint callbacks were employed to monitor validation performance and prevent overfitting.
   - The model was trained on 90% of the data and validated on the remaining 10%.

5. **Model Evaluation:**
   - The model's performance was evaluated on the test set, assessing accuracy and loss.
   - Training and validation loss curves were plotted to visualize the learning process and detect potential overfitting.

6. **Age Prediction:**
   - A function was developed to predict the age group of new images.
   - The function preprocesses the input image, makes predictions using the trained model, and maps the predictions to age groups.

## Visualization of the model used
![0_cV6Ciyjm0pdebW_2-ezgif com-webp-to-jpg-converter](https://github.com/ThisaraWeerakoon/Age-Classification/assets/83450623/e3bf3776-6907-4240-987d-5707abcb6ee9)

## Code Implementation

The project's code is organized in a Jupyter notebook, which includes detailed steps for data preprocessing, model training, and evaluation. Key libraries used in the project include:

- `numpy` for numerical operations
- `matplotlib` for data visualization
- `cv2` (OpenCV) for image processing
- `keras` for building and training the neural network
- `visualkeras` for visualizing the model architecture

## Example Usage

To test the trained model on new images, follow these steps:

1. **Preprocess the Image:**
   ```python
   def image_preprocessing(img_path):
       img = cv2.imread(img_path)
       resized_img = cv2.resize(img, (224, 224))
       normalized_img = resized_img / 255.0
       return normalized_img
2.**Predict Age Group:**
 ```python
  def predict_on_image(img_path):
    preprocessed_img = image_preprocessing(img_path)
    reshaped_img = np.reshape(preprocessed_img, (1, 224, 224, 3))
    predicted_labels_probabilities = model.predict(reshaped_img)
    class_index = np.argmax(predicted_labels_probabilities)
    age_class = str(class_index * 25) + "-" + str((class_index + 1) * 25 - 1)
    return age_class
```

3.**Visualize Prediction:**
 ```python
  new_sample_img_rgb = cv2.cvtColor(new_sample_img_bgr, cv2.COLOR_BGR2RGB)
  cv2.putText(new_sample_img_rgb, predicted_age_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
  plt.imshow(new_sample_img_rgb)
```

## Credits

We used several third-party assets and tutorials, including:

- [Tensorflow](https://www.tensorflow.org/api_docs)
- [VGG16 Model](https://keras.io/api/applications/vgg/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Badges

![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
	![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## How to Contribute

We welcome contributions from the community! If you are interested in contributing, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
    ```sh
    git checkout -b feature-or-bugfix-name
    ```
3. Commit your changes:
    ```sh
    git commit -m "Description of the feature or bug fix"
    ```
4. Push to the branch:
    ```sh
    git push origin feature-or-bugfix-name
    ```
5. Open a pull request and provide a detailed description of your changes.
