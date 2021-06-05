# Contactless_Covid_Tester_Web_App
![](https://assets.entrepreneur.com/content/3x2/2000/20200331145028-coronavirus-4967489-1920.jpeg?width=1000)

### Inspiration
COVIID-19, the most exasperating topic spoken all across the Globe. We are all aware of this virus and its consequences. People get afraid due to the symptoms of COVID-19 and crowd at the hospitals which is currently very difficult for the frontline workers to manage and take tests for all. It is very important to know who actually needs to be tested. Hence people who are at a higher risk are only needed to be targeted.

Also it has been found that the best medicine for COVID-19 is improving our immunity. So, a personalized diet suggestions for encountering COVID would improve each and every one to win this battle against COVID.

Lot of fake news about COVID are spreading on social Media where people get confused on what to do. So it would be better if they get answers for their questions on what to do and what not to do.

### How this application works
A simple Web App integrated with AI systems is implemented which helps the Doctor to diagnose the patients without visiting them. The Web App contains the following:

* Contactless Test
* Personalized Diet Suggestions for COVID-19
* Chatbot

### Contactless Test
A risk score is generated by the AI system which ranges from 1 - 100. It is this score that helps to prioritize the patients. Higher the risk score, higher is the importance given to that patient.

The Contactless test consists of the following:

### DRY COUGH SEVERITY
The user is made to cough and its sound is recorded and sent to the Database. The Convolutional Neural Networks reads the input frequencies using the spectrogram images and predicts the dry cough rate based on how severe the dry cough is. If its a normal cough or a wet cough, the dry cough rate would be less. This test contributes to the risk sore.

### HEART BPM | RESPIRATORY TEST
The user is made to take deep breath in and out for 15 - 20 seconds which is recorded by the camera. The Convolutional Neural Networks read the input image sequence and generates the respiratory rate and bpm. Based on these values, the AI system keeps a track to the risk score.

### QUERIES
Well framed queries are questioned where the user need to answer. Eg: "What is your Current Location?", Do you have any travel history in the past 30 days?, Are you able to smell odorous items? etc..,

The AI system itself finds how severe COVID is affected in the user's current location and if its high, then accordingly it gets tracked to the risk score. In the similar way all other questions are also tracked by the AI system to the risk score. Finally the AI System generates a risk score for the patient. Now Doctors can login the Web App at any time and can diagnose all the patients based on these tests. Finally the Doctor writes the prescription/remarks for the patient.This is sent to the patient's mail. As a prioritization is made, highly risky patients are diagnosed first. The user can follow the guidelines as prescribed by the Doctor in the Prescription.

### AI Personalized Diet Suggestions for COVID-19
The users need to answer the questions which helps to calculate the magnesium, zinc,fat,water,sugar and vitamin content in their body. This is sent to the database. Random Forest Algorithm is used to determine what kind of diet is needed for that person in order to improve the immune system.

### Chatbot
Its an RNN based chatbot. People can ask any kind of queries regarding COVID. All the data's and facts about COVID were fetched from official websites like WHO. So people can ask like "How to go to a grocery shop?", "Should I use masks at home?", "Are kids affected equally ?" etc...,

### How this application was developed
* The COVID Test, Chatbot, Diet Suggestions were created using the tensorflow and keras frameworks. (Python)
* The UI was entirely made using HTML, CSS, Javascript.
* Three MYSQL Database was made locally using phpMyAdmin(Patient,Doctor,Patient's Record).
* The AI system, UI and MYSQL was integrated using flask.

### Requiremenmts
* python 3.6+
* Flask
* sqlalchemy
* MySQL
* soundfile
* numpy
* shutil
* fpdf
* nltk
* tflearn
* tensorflow
* smtplib
* sklearn
* keras
* cv2
* imutils
* time
* face_utilities
* librosa
* pandas

### How to run the app:
1. First connect with your phpmyadmin and create a database name covid and create 3 table(doctor,patient,record) with (first_name,last_name,email,password,mobile) , (first_name,last_name,email,password) , (id,name,email,cough,bpm,resp,risk) as columns.

2.Clone this repository .
```
https://github.com/karthikeyanthanigai/Contactless_Covid_Tester_Web_App.git
```
3.Open command line and set the directory to the cloned repository.
```
cd Contactless_Covid_Tester_Web_App/flask_app
```
4.Enter the command.
```
python app.py
```

If you got any error in installing the packages then feel free to refer [Stackoverflow](https://www.stackoverflow.com).

# Links:
### Youtube:https://youtu.be/0LsfJNB12uE
