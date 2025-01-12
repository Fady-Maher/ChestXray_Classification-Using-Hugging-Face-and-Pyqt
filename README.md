# ChestXray_Classification-Using-Hugging-Face-and-Pyqt
Chest X-rays (CXRs) are among the most commonly used diagnostic tools in medical practice, playing an essential role in detecting various lung and thoracic conditions, including pneumonia, normal lung states, COVID-19, and lung cancer. Despite their utility, interpreting CXRs requires substantial expertise, and even seasoned radiologists may struggle to discern subtle differences indicative of specific conditions.<br>
Large Language Models (LLMs) have emerged as powerful tools to enhance the diagnostic process by enabling precise and efficient classification of CXR images.
Problem: A major challenge in automated CXR classification is the potential failure when the input image is not a chest X-ray.<br>
Solution: To address this, we integrate another dataset containing approximately 80 object categories with the chest X-ray dataset. A combined model (Model 1) is then trained on this expanded dataset to classify whether an input image is a chest X-ray or not.
Features
•	CXR Classification: Detects and categorizes chest X-rays into various diagnostic conditions ( ).
•	Input Image Validation: Differentiates chest X-ray images from other types of images to prevent errors.
References: <br>
•	Chest X-ray Pneumonia Dataset :  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia<br>
•	COVID-19 Radiography Database : https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database<br>
Tools and Technologies:<br>
•	Python: For backend processing and model development.<br>
•	Keras/TensorFlow: For training and evaluation of deep learning models.<br>
•	Hugging Face: For pre-trained model integration and fine-tuning.<br>
•	PyQt: To build a graphical user interface for interaction.<br>



