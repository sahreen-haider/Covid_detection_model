# Covid_detection_model

## This is a Custom model built from scratch, 
## The aim is to detect if a person has covid using CT-Scans,

### The model includes 4 Convolution layer
### 2 fully connected layers

### The model is trained on 4 Different classes of Data:
### Classes: Covid, Healthy, Others
1. Covid directory contains CT_scan Samples of Various Patients with Covid.
2. Healthy Directory contains Ct_scan samples of  various Healthy patients.
3. Others Directory containts CT_scan samples of patients with other disease.

The repository packs a requirements.txt file that has all the packages and modules listed to implement the model/project
from the strach.


### The directory structure looks like this:
|-- ct0   Virtual env directory
|-- CT_covid_classificatin_models.ipynb
|-- rename_directories.py
|-- requirements.txt
|-- weights and model architecture is saved on model.h5

### The POC is Deployed using streamlit

you can use upload scan and get results for the particular,
--> Supported images are 'jpg', 'jpeg', 'png'
--> Support for .MHA and .NII files will addded in the future

#### For predicting or using your own model or architecture
Use your own weights and architecture file using saved model file
formats suppported or "h5py" and "json"

## The below is a CT-Scan sample of a person with covid.
![Anota0](https://github.com/sahreen-haider/Covid_detection_model/assets/81517526/ffc52414-cffa-447b-885d-90eb64c8271d)

## The below is a CT-Scan sample of a healthy person.
![Anota0](https://github.com/sahreen-haider/Covid_detection_model/assets/81517526/1b253510-7acb-4ee9-a80a-24e13c2749f5)

## The below is a CT-Scan sample of person with other diseases.
![Anota0](https://github.com/sahreen-haider/Covid_detection_model/assets/81517526/afe864bc-fd7b-4e97-935a-9c38a2196104)

## To use the POC.
navigate to the directory and run the below command in your terminal,
--> $ streamlit run deployed_pro.py


# The below is a link to the dataset:
https://drive.google.com/drive/folders/1t6QUfsdgY1hVEVdnGMeouJ7dCY8Gf6LE?usp=sharing
