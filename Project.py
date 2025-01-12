#librarys for the script python
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
from PyQt5 import QtGui
import torch
import timm
import torch.nn as nn
import time
from os import path
import cv2
import numpy as np
from transformers import AutoModelForImageClassification, AutoProcessor
from torchvision import transforms
from PIL import Image
from transformers import MarianMTModel, MarianTokenizer



classes_chest = {0:'Covid_19',1:'Lung Opacity',2:'Normal',3:'Pheumonia'}
classes_not = {0:'Chest X-ray',1:'Not Chest Xray'} 

import openai

# Set your OpenAI API key
#openai.api_key = ''  # Replace with your OpenAI API key

def generate_text(prompt):
    try:

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can also use "gpt-4" or other models
            messages=[{"role": "user", "content": prompt}],  # The input message
            max_tokens=100,  # Maximum tokens in the response
            temperature=0.7  # Controls randomness (0.0 is deterministic, 1.0 is more random)
        )

        result = response['choices'][0]['message']['content'].strip()
        return result

    except openai.error.OpenAIError as e:
        return f"Error: {e}"
    
LANGUAGE_MODELS = {
    "English to Spanish": "Helsinki-NLP/opus-mt-en-es",
    "English to French": "Helsinki-NLP/opus-mt-en-fr",
    "English to Hungarian": "Helsinki-NLP/opus-mt-en-hu",
    "English to Arabic": "Helsinki-NLP/opus-mt-en-ar",
}

def load_model_tr(input_text, lang_pair):
    try:
        model_name = LANGUAGE_MODELS.get(lang_pair)        
        if not model_name:
            raise ValueError(f"Model for language pair '{lang_pair}' not found.")
        print("Loading model for:", lang_pair)
        
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        inputs = tokenizer([input_text], return_tensors="pt", truncation=True, padding=True)
        translated = model.generate(**inputs)
        
        output = tokenizer.decode(translated[0], skip_special_tokens=True)
        return output
    
    except Exception as e:
        print(f"Error occurred during translation: {e}")
        return None

def predict_image(image_path,processor,model,transform):
    try:
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert("RGB")
        img = transform(img)
        img = torch.clamp(img, 0, 1)
        inputs = processor(images=img, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()  # Get the predicted class index
        return predicted_class
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def load_model_xray_not():
    model_path = "vit_all_classifier"
    model = AutoModelForImageClassification.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    return model, processor, transform
    
def load_model_xray():
    model = timm.create_model("hf_hub:timm/mobilenetv3_large_100.ra_in1k", pretrained=False, num_classes=4)  # Use pretrained=False since you're loading custom weights
    model.load_state_dict(torch.load('mobilenetv3_finetuned.pth', map_location=torch.device('cpu')))
    model.to(torch.device('cpu'))
    model.eval()
    print("Model loaded and moved to CPU.")
    return model

def predict_class_xray(image_path, model):
    print("predict_class_xray : ", image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

    image = Image.open(image_path).convert('RGB')  
    image = transform(image).unsqueeze(0)  
    image = image.to(torch.device('cpu'))
    # Make a prediction
    with torch.no_grad():  
        outputs = model(image) 
        # Get the index of the class with the highest score
        _, predicted_class = torch.max(outputs, 1)  
        print("predicted_class.item()   : ",predicted_class.item()   , "   \n\n" )
    return predicted_class.item()  

def read_data_fun(update_ui,imagePath,ln):
     update_ui(percent=25,predict_class1="Contain of image is being identified"
                                   ,predict_class2 ="The Chest X-Ray is being identified", desc="",translation="") 
    
     model_not, processor_not, transform_not = load_model_xray_not()
     predict1 = predict_image(imagePath,processor_not,model_not,transform_not)
     
     update_ui(percent=25,predict_class1="Contain of image is being identified"
                                    ,predict_class2 ="The Chest X-Ray is being identified", desc="",translation="") 
     
     if predict1 == 0 :
       model_xray = load_model_xray()
       predict2 = predict_class_xray(imagePath,model_xray)
       print("chest xRAY Disease : ",predict2)
       desc = "Provide a detailed description of [ " +classes_chest.get(predict2) +" ], including :   its causes, common symptoms, diagnosis methods, available treatments, and potential complications. Provide a clear, medically accurate explanation suitable for a general audience"
     else:
       print("imag doesn not contain chest xray!!")
       predict2 = " Not Known"
       desc = "Provide a clear, accurate explanation that image is not contain chest x ray this  explanation should be suitable for a general audience"
     desc = generate_text(desc)

     time.sleep(2)
     update_ui(percent=50,predict_class1="Contain of image is being identified"
                                    ,predict_class2 ="The Chest X-Ray is being identified", desc="",translation="")  

     translation = load_model_tr(desc, ln)
 
        
     time.sleep(2)
     update_ui(percent=50,predict_class1="Contain of image is being identified",predict_class2 =  "The Chest X-Ray is being identified" , desc="",translation="")  
     time.sleep(3)
     update_ui(percent=70,predict_class1=predict1,predict_class2 ="The Chest X-Ray is being identified" , desc = "....", translation ="......")   
     time.sleep(1)
     update_ui(percent=80 ,predict_class1=predict1, predict_class2 = predict2 ,desc=  desc,translation= translation) 
     time.sleep(1)
     update_ui(percent=90 ,predict_class1 = predict1,predict_class2 =predict2 ,desc = desc,translation =translation) 
     time.sleep(1)
     update_ui(percent=100,predict_class1 =predict1,predict_class2 =predict2 , desc =desc,translation =translation) 
 
class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int,int,int,str,str)
    def intail(self, path_image,ln):
      self.path_image = path_image
      self.ln = ln
    
    def control_progessbar(self):
         read_data_fun(self.control_update_progress,self.path_image,self.ln)
         self.finished.emit()
             
    def control_update_progress(self,percent =0 ,predict_class1="",predict_class2="" 
                                , desc="",translation=""):
         print('percent',percent , "   + predict_class2 : ", predict_class2)
         self.progress.emit(percent,predict_class1,predict_class2 , desc,translation) 
    
FORM_CLASS,_ = loadUiType(path.join(path.dirname(__file__),"Classification_ui.ui"))

class MainApp(QMainWindow , FORM_CLASS):
    def __init__(self , parent=None):
        super(MainApp , self).__init__(parent)
        QMainWindow.__init__(self)
        icon = QIcon()
        icon.addPixmap(QPixmap("logo_new.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        self.setWindowIcon(icon)
        self.setWindowTitle("Classification of Chest X-ray")
    
        self.new_width = 220
        self.new_height = 170
        self.action = False
        self.imagePath = ''
        self.white_pixmap = QPixmap( self.new_width,self.new_height)
        self.white_pixmap.fill(QColor("white"))
        self.setupUi(self)
        self.handle_buttons()
        self.btnPredict.setEnabled(False)
        self.comboBox.addItems(LANGUAGE_MODELS.keys())
            
             
    def handle_buttons(self):
        self.btnBrowseImage.clicked.connect(self.getImage)
        self.btnPredict.clicked.connect(self.getPredict)


    def disable_buttons(self):
        print('disable_buttons')
        self.btnBrowseImage.setEnabled(False)
        self.btnPredict.setEnabled(False)

    def enable_buttons(self):
         print('enable_buttons')
         self.btnBrowseImage.setEnabled(True)
         self.btnPredict.setEnabled(True)    

    def getImage(self):
        self.reset_component()
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files (*.jpg *.png)")
        self.disable_buttons()
        self.imagePath = fname[0]
        print("imagePath: ",self.imagePath)
        pixmap = QPixmap(self.imagePath)
        resized_pixmap = pixmap.scaled(self.new_width, self.new_height)
        self.labelImage1.setPixmap(QPixmap(resized_pixmap))
        if len(fname[0])>0:
            self.btnPredict.setEnabled(True) 
        self.btnBrowseImage.setEnabled(True)

        
    def getPredict(self): 
        ln = self.comboBox.currentText()
        print("=== : ",self.comboBox.currentText())
        self.disable_buttons()
        self.read_data_update_progress(percent=0,predict_class1="Contain of image is being identified"
                                       ,predict_class2 ="The Chest X-Ray is being identified", desc="",translation="") 
        self.thread = QThread()
        self.worker = Worker()
        self.worker.intail(self.imagePath,self.comboBox.currentText())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.control_progessbar)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.read_data_update_progress)
        self.thread.start()
         
    def read_data_update_progress(self,percent,predict_class1,predict_class2 , desc,translation):
        self.progressBar.setValue(percent)
        self.label_des.setWordWrap(True)  # Enable word wrap
        self.label_tr.setWordWrap(True)  # Enable word wrap
        self.label_des.setStyleSheet("""
            QLabel {
                padding: 10px 10px, 1px 5px;  /* top-bottom, left-right */
                border: 1px solid black;
                background-color: white;  /* Set the background color to white */
            }
        """)
        
        self.label_tr.setStyleSheet("""
            QLabel {
                padding: 10px 10px, 5px 1px;  /* top-bottom, left-right */
                border: 1px solid black;
                background-color: white;  /* Set the background color to white */
            }
        """)

        if percent == 100:
            self.label_new.setText("Complete")
            self.action = True
         
            self.label_tr.setText("Translation "+str(translation))
            self.enable_buttons()
            print("successful")
        elif percent == 90:
            self.label_new.setText("Finalizing Results")
            self.label_des.setText("Description of Diseases : " + str(desc))
            
        elif percent == 80:
            self.labelPredict1.setText("The Type of diseases is  : " + str(classes_chest.get(predict_class2)))
        elif percent == 70:
            self.labelPredict2.setText("The Images contains : " + str(classes_not.get(predict_class1)))
        elif percent == 40:
            self.label_new.setText("Running Classification Model")
        elif percent == 50:
            self.label_new.setText("Preprocessing Image")
        elif percent == 25:
            self.label_new.setText("Loading Image")
            
    def reset_component(self):
        self.action =False
        self.progressBar.setValue(0)
        self.label_des.setText("")
        self.label_tr.setText("")
        self.labelPredict1.setText("")
        self.labelPredict2.setText("")

def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()