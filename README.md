# ChestXray_Classification-Using-Hugging-Face-and-Pyqt
Chest X-rays (CXRs) are among the most commonly used diagnostic tools in medical practice, playing an essential role in detecting various lung and thoracic conditions, including pneumonia, normal lung states, COVID-19, and lung cancer. Despite their utility, interpreting CXRs requires substantial expertise, and even seasoned radiologists may struggle to discern subtle differences indicative of specific conditions.
Large Language Models (LLMs) have emerged as powerful tools to enhance the diagnostic process by enabling precise and efficient classification of CXR images.
Problem: If the input image is not a chest X-ray, the diagnostic process can fail.
Solution: To address this, we integrate another dataset containing approximately 80 object categories with the chest X-ray dataset. A combined model (Model 1) is then trained on this expanded dataset to classify whether an input image is a chest X-ray or not.
References:
•	Chest X-ray Pneumonia Dataset :  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
•	COVID-19 Radiography Database : https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

