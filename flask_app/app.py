from flask import Flask, render_template, request
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from flask_mysqldb import MySQL
import re
import soundfile as sf

# Importing the libraries
import numpy as np
import shutil
import pandas as pd

from fpdf import FPDF
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import nn
import sys
import keras

import cv2
import numpy as np
import time
import sys
from imutils import face_utils
from face_utilities import Face_utilities
from signal_processing import Signal_processing
count = 0
rates = []
email1=''
acc=''
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

app = Flask(__name__)

app.config['MYSQL_HOST' ]='::1'
app.config['MYSQL_USER']='root'
app.config['MYSQL_PASSWORD']= ''
app.config['MYSQL_DB']='covid'
mysql=MySQL(app)







def bpmrpm():
    count = 0
    rates = []
    video = True

    if video == False:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture("video.webm")

    fu = Face_utilities()
    sp = Signal_processing()

    i=0
    last_rects = None
    last_shape = None
    last_age = None
    last_gender = None

    face_detect_on = False
    age_gender_on = False

    t = time.time()

    #for signal_processing
    BUFFER_SIZE = 100

    fps=0 #for real time capture
    video_fps = cap.get(cv2.CAP_PROP_FPS) # for video capture
    #print(video_fps)

    times = []
    data_buffer = []

    # data for plotting
    filtered_data = []

    fft_of_interest = []
    freqs_of_interest = []

    bpm = 0



    def update():
        p1.clear()
        p1.plot(np.column_stack((freqs_of_interest,fft_of_interest)), pen = 'g')

        p2.clear()
        p2.plot(filtered_data[20:],pen='g')
        app.processEvents()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(300)

    while True:
        # grab a frame -> face detection -> crop the face -> 68 facial landmarks -> get mask from those landmarks

        # calculate time for each loop
        t0 = time.time()

        if(i%1==0):
            face_detect_on = True
            if(i%10==0):
                age_gender_on = True
            else:
                age_gender_on = False
        else:
            face_detect_on = False

        ret, frame = cap.read()
        #frame_copy = frame.copy()

        if frame is None:
            #print("End of video")
            cv2.destroyAllWindows()
            timer.stop()
            #sys.exit()
            break

        #display_frame, aligned_face = flow_process(frame)


        ret_process = fu.no_age_gender_face_process(frame, "68")

        if ret_process is None:
            cv2.putText(frame, "No face detected", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
            cv2.imshow("frame",frame)
            #print(time.time()-t0)

            cv2.destroyWindow("face")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                timer.stop()
                #sys.exit()
                break
            continue

        rects, face, shape, aligned_face, aligned_shape = ret_process

        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        #overlay_text = "%s, %s" % (gender, age)
        #cv2.putText(frame, overlay_text ,(x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)

        if(len(aligned_shape)==68):
            cv2.rectangle(aligned_face,(aligned_shape[54][0], aligned_shape[29][1]), #draw rectangle on right and left cheeks
                    (aligned_shape[12][0],aligned_shape[33][1]), (0,255,0), 0)
            cv2.rectangle(aligned_face, (aligned_shape[4][0], aligned_shape[29][1]),
                    (aligned_shape[48][0],aligned_shape[33][1]), (0,255,0), 0)
        else:
            #print(shape[4][1])
            #print(shape[2][1])
            #print(int((shape[4][1] - shape[2][1])))
            cv2.rectangle(aligned_face, (aligned_shape[0][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
                        (aligned_shape[1][0],aligned_shape[4][1]), (0,255,0), 0)

            cv2.rectangle(aligned_face, (aligned_shape[2][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
                        (aligned_shape[3][0],aligned_shape[4][1]), (0,255,0), 0)

        for (x, y) in aligned_shape:
            cv2.circle(aligned_face, (x, y), 1, (0, 0, 255), -1)


        #for signal_processing
        ROIs = fu.ROI_extraction(aligned_face, aligned_shape)
        green_val = sp.extract_color(ROIs)
        #print(green_val)

        data_buffer.append(green_val)

        if(video==False):
            times.append(time.time() - t)
        else:
            times.append((1.0/video_fps)*i)

        L = len(data_buffer)
        #print("buffer length: " + str(L))

        if L > BUFFER_SIZE:
            data_buffer = data_buffer[-BUFFER_SIZE:]
            times = times[-BUFFER_SIZE:]
            #bpms = bpms[-BUFFER_SIZE//2:]
            L = BUFFER_SIZE
        #print(times)
        if L==100:
            fps = float(L) / (times[-1] - times[0])
            cv2.putText(frame, "fps: {0:.2f}".format(fps), (30,int(frame.shape[0]*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            #
            detrended_data = sp.signal_detrending(data_buffer)
            #print(len(detrended_data))
            #print(len(times))
            interpolated_data = sp.interpolation(detrended_data, times)

            normalized_data = sp.normalization(interpolated_data)

            fft_of_interest, freqs_of_interest = sp.fft(normalized_data, fps)

            max_arg = np.argmax(fft_of_interest)
            bpm = freqs_of_interest[max_arg]
            cv2.putText(frame, "HR: {0:.2f}".format(bpm), (int(frame.shape[1]*0.8),int(frame.shape[0]*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            #print(detrended_data)
            filtered_data = sp.butter_bandpass_filter(interpolated_data, (bpm-20)/60, (bpm+20)/60, fps, order = 3)
            #print(fps)
            #filtered_data = sp.butter_bandpass_filter(interpolated_data, 0.8, 3, fps, order = 3)

        #write to txt file
        with open("a.txt",mode = "a+") as f:

            rates.append(bpm)
            f.write("time: {0:.4f} ".format(times[-1]) + ", HR: {0:.2f} ".format(bpm) + "\n")

        # display

        #cv2.imshow("mask",mask)
        i = i+1
        #print("time of the loop number "+ str(i) +" : " + str(time.time()-t0))
        count = count + 1


    rates = [i for i in rates if i!=0]
    #print(rates)

    avg = sum(rates)/len(rates)
    resp = avg/4.5
    print("Heart Rate: ",avg)
    print("Respiratory Rate: ", resp)
    l=[]
    l.append(avg)
    l.append(resp)
    return l
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/main")
def main():
    return render_template("final.html",account=acc)
@app.route("/blogin")
def blogin():
    return render_template("blogin.html")
@app.route("/test")
def test():
    return render_template("test1.html")
@app.route("/record")
def record():
    return render_template("record.html")
@app.route("/cof")
def cof():
    return render_template("cof.html")
@app.route("/testresult")
def testresult():
    return render_template("testresult.html")

@app.route('/docr', methods = ['POST'])
def docr():
    k=request.form['Respond']
    cur = mysql.connection.cursor()

    cur.execute("SELECT * FROM record where id = %s", (k,))
    data = cur.fetchone()
    return render_template("pre.html",data=data)


@app.route('/mail', methods = ['POST'])
def mail():
    k=request.form['Respond']
    kk=request.form['text']
    cur = mysql.connection.cursor()

    cur.execute("SELECT * FROM record where id = %s", (k,))
    data = cur.fetchone()

    pdf = FPDF()

    # Add a page
    pdf.add_page()

    # set style and size of font
    # that you want in the pdf
    pdf.set_font("Arial", size = 30)

    # create a cell
    pdf.cell(200, 10, txt = "VirTest",
    		ln = 1, align = 'L')

    # add another cell
    pdf.cell(200, 10, txt = "www.virtest.com",
    		ln = 2, align = 'L')
    pdf.cell(200, 10, txt = "",
    		ln = 2, align = 'L')
    pdf.cell(200, 10, txt = "Dr.Jason Wick                  Date:15/06/2020",
    		ln = 2, align = 'L')
    pdf.cell(250, 10, txt = "-----------------------------------------------------",
    		ln = 2, align = 'L')
    pdf.cell(200, 10, txt = "",
    		ln = 2, align = 'L')




    pdf.cell(200, 10, txt = "Name:"+str(data[1])+"                        Age:45",
    		ln = 2, align = 'L')
    pdf.cell(200, 10, txt = "Heart BPM:"+str(data[4])+"        Respiratory Rate:"+str(data[5]),
    		ln = 2, align = 'L')

    pdf.cell(250, 10, txt = "Cough Severity:"+str(data[3]) ,
    		ln = 2, align = 'L')
    pdf.cell(250, 10, txt = "-----------------------------------------------------",
    		ln = 2, align = 'L')
    pdf.cell(250, 10, txt = "Remarks:",
    		ln = 2, align = 'L')
    pdf.cell(200,10,txt=str(kk),ln = 2, align = 'L')



    # save the pdf with name .pdf
    pdf.output("VirtTest_Prescription.pdf")

    fromaddr = "doctor@gmail.com"
    toaddr = "patient@gmail.com"

    # instance of MIMEMultipart
    msg = MIMEMultipart()

    # storing the senders email address
    msg['From'] = fromaddr

    # storing the receivers email address
    msg['To'] = toaddr

    # storing the subject
    msg['Subject'] = "Just Testing"

    # string to store the body of the mail
    body = "Reply from VirTest. Thanks for taking Test at VirTest. Please go through the Attachment for Doctor's Prescription"

    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    # open the file to be sent
    filename = "VirtTest_Prescription.pdf"
    attachment = open("VirtTest_Prescription.pdf", "rb")

    # instance of MIMEBase and named as p
    p = MIMEBase('application', 'octet-stream')

    # To change the payload into encoded form
    p.set_payload((attachment).read())

    # encode into base64
    encoders.encode_base64(p)

    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    # attach the instance 'p' to instance 'msg'
    msg.attach(p)

    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)

    # start TLS for security
    s.starttls()

    # Authentication
    s.login(fromaddr, "Password_of_Sender")

    # Converts the Multipart msg into a string
    text = msg.as_string()

    # sending the mail
    s.sendmail(fromaddr, toaddr, text)

    # terminating the session
    s.quit()

    return str(kk)

@app.route('/messages', methods = ['POST'])
def api_message():
    f = open('./file.wav', 'wb')
    f.write(request.data)
    f.close()


    data, samplerate = sf.read('file.wav')
    sf.write('new_file.ogg', data, samplerate)



@app.route("/restest" , methods=['GET','POST'])
def restest():
    one = request.form.get('gender')
    two = int(request.form.get('age'))
    three = request.form.get('location')
    four = request.form.get('dia')
    five = request.form.get('head')
    six = request.form.get('odour')
    seven = request.form.get('sym')
    eight= request.form.get('fever')
    nine=request.form.get('dis')
    ten=request.form.get('bp')
    ele=request.form.get('trv')


    j=[one,two,three,four,five,six,seven,eight,nine,ten]
    states = ["Tamil Nadu", "Maharashtra", "Delhi","Gujarat","Uttar Pradesh","Rajasthan","Madhya Pradesh","West Bengal","Karnataka","Bihar","Haryana","Andhra Pradesh","Jammu and Kashmir","Telangana","Odisha","Assam","Punjab","Kerala","Jharkhand","Uttarakhand","Chhattisgarh","Tripura","Himachal Pradesh","Goa","Manipur","Puducherry","Ladakh","Nagaland","Mizoram","Arunachal Pradesh","Meghalaya","Andaman and Nicobar Islands","Sikkim","Lakshadweep","New York","New Jersey","California","Illinois","Massachusetts","Texas","Pennsylvania","Michigan","Florida","Maryland","Georgia","Virginia","Connecticut","Louisiana","North Carolina","Ohio","Indiana","Arizona","Minnesota","Colorado","Tennessee","Washington","Wisconsin","lowa","Alabama","Mississippi","South Carolina","Rhode island","Nebraska","Missouri","Utah","Kansas","Kentucky","Delaware","New Mexico","Arkansas","Washington,D.c","Nevada","Oklahoma","South Dakota","New Hampshire","Oregon","Puerto Rico","Idaho","North Dakota","Maine","West Virginia","Vermont","Wyoming","Hawai","Montana","Alaska","Guam","U.S.Virgin islands","Northern Mariana islands","American Samoa"]
    cases=["38716","97648","34687","22032","12088","12076","10241","9768","6245","5983","5968","5429","4574","4320","3498","3464","2969","2327","1599","1411","1398","913","470","417","366","157","135","128","102","61","44","38","14","0","381000","166000","133000","131766","104000","83409","77780","64998","66602","59136","54242","51721","44092","43492","40791","39266","37623","31264","29763","28183","24375","22484","21641","2,015","19614","18109","15228","14991","14611","13767","12864","10867","10315","9773","9367","9549","9016","8935","6676","5095","4876","4576","3935","2975","2625","2512","2071","1042","903","653","525","487","169","73","22","0"]


    #cought
    le = LabelEncoder()
    prob = nn.predict("new_file.ogg","trained_cnn.h5")
    print(prob)


    #video
    a=bpmrpm()

    x=1.90
    total=0
    if two > 50:
        total=total+2
    if one=='male':
        total+=1.2
    if ten=='yes':
        total+=2
    if nine=='yes':
        total+=3
    if four=='yes':
        total+=3
    threee=states.index(three)
    count=int(cases[threee])
    if count<1000:
        total+=2
    elif count<3000:
        total+=3
    elif count<10000:
        total+=4
    else:
        total+=5

    if ele=='yes':
        total+=3
    if five=='yes':
        total+=1
    if seven=='yes':
        total+=2
    if six=='no':
        total+=2
    if eight=='yes':
        total+=2
    test=0
    total=total*x
    if a[0]<60 or a[0]>100:
        test+=100
    if prob >80:
        test+=100
    elif prob>60:
        test+=75
    elif prob>50:
        test+=50
    test=test/4

    total=total+test
    print(test,total)

    cur = mysql.connection.cursor()
    cur.execute('INSERT INTO record(name,email,cough,bpm,resp,risk) VALUES (%s,%s,%s,%s,%s,%s);', (acc,email1,prob,a[0],a[1],total,))
    mysql.connection.commit()
    cur.close()

    return render_template("testresult.html",cou=prob,res=a[1],bp=a[0],risk=total,email=email1,name=acc)
@app.route("/bdiet")
def bdiet():
    return render_template("bdiet.html")
@app.route("/veg")
def veg():
    return render_template("veg.html",data=[{'val':0}, {'val':1}, {'val':2}, {'val':3}, {'val':4}, {'val':5}, {'val':6}, {'val':7}, {'val':8}, {'val':9}, {'val':10}])
@app.route("/vtest" , methods=['GET','POST'])
def vtest():
    one = int(request.form.get('one'))
    two = int(request.form.get('two'))
    three = int(request.form.get('three'))
    four = int(request.form.get('four'))
    five = int(request.form.get('five'))
    six = int(request.form.get('six'))
    seven = int(request.form.get('seven'))
    eight = int(request.form.get('eight'))
    nine = int(request.form.get('nine'))
    v=[one,two,three,four,five,six,seven,eight,nine]
    print(v)
    mag=(three+five+four+six)/4
    zin=(three+six+four)/3
    swe=(two+seven)/2
    wat=(one+eight)/2
    fat=(four+nine+six)/3
    f=[mag,zin,swe,wat,fat]
    # Random Forest Classification
    print(f)


    # Importing the dataset
    dataset = pd.read_csv('Book.csv')
    X = dataset.iloc[:, 0:5].values
    y = dataset.iloc[:, -1].values

    # Encoding categorical data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder = LabelEncoder()
    y[:] = labelencoder.fit_transform(y[:])
    y=y.astype('int')

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


    # Training the Random Forest Classification model on the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)



    kk = np.asarray(f)
    kk=kk.reshape(1, -1)
    y_pred1 = classifier.predict(kk)
    a = y_pred1.tolist()
    print(a)
    if a[0]==0:
        return render_template("type1.html")
    if a[0]==1:
        return render_template("type2.html")
    if a[0]==2:
        return render_template("type3.html")
    if a[0]==3:
        return render_template("type4.html")
    if a[0]==4:
        return render_template("type5.html")
    if a[0]==5:
        return render_template("type6.html")
    if a[0]==6:
        return render_template("type7.html")
@app.route("/nonveg")
def nonveg():
    return render_template("nonveg.html",data=[{'val':0}, {'val':1}, {'val':2}, {'val':3}, {'val':4}, {'val':5}, {'val':6}, {'val':7}, {'val':8}, {'val':9}, {'val':10}])
@app.route("/nvtest" , methods=['GET','POST'])
def nvtest():
    one = int(request.form.get('one'))
    two = int(request.form.get('two'))
    three = int(request.form.get('three'))
    four = int(request.form.get('four'))
    five = int(request.form.get('five'))
    six = int(request.form.get('six'))
    seven = int(request.form.get('seven'))
    eight = int(request.form.get('eight'))
    nine = int(request.form.get('nine'))
    ten = int(request.form.get('ten'))
    ele = int(request.form.get('ele'))
    twl= int(request.form.get('twl'))
    the = int(request.form.get('the'))
    v=[one,two,three,four,five,six,seven,eight,nine,ten,ele,twl,the]
    print(v)

    mag=(three+five+four+six+ele+the)/6
    zin=(three+six+four+twl+ele)/5
    swe=(two+seven)/2
    wat=(one+eight)/2
    fat=(four+nine+six+ten+twl+the)/6
    fi=[mag,zin,swe,wat,fat]
    # Random Forest Classification
    print(fi)
    # Random Forest Classification



    # Importing the dataset
    dataset = pd.read_csv('Book.csv')
    X = dataset.iloc[:, 0:5].values
    y = dataset.iloc[:, -1].values

    # Encoding categorical data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder = LabelEncoder()
    y[:] = labelencoder.fit_transform(y[:])
    y=y.astype('int')

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


    # Training the Random Forest Classification model on the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)



    kk = np.asarray(fi)
    kk=kk.reshape(1, -1)
    y_pred1 = classifier.predict(kk)
    aa = y_pred1.tolist()
    print(aa)

    if aa[0]==0:
        return render_template("type1.html")
    if aa[0]==1:
        return render_template("type2.html")
    if aa[0]==2:
        return render_template("type3.html")
    if aa[0]==3:
        return render_template("type4.html")
    if aa[0]==4:
        return render_template("type5.html")
    if aa[0]==5:
        return render_template("type6.html")
    if aa[0]==6:
        return render_template("type7.html")


    #return render_template("vtest.html")
@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/chatbottest", methods=["POST"])
def chatbottest():
    val = str(request.form.get("val"))

    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None, 97])
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 10, activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    model.load("model.tflearn")

    labels=['COVID', 'children', 'grocery', 'isloate', 'nospread', 'protection', 'spread', 'symptoms', 'vaccine', 'virus']

    words=[',', 'a', 'about', 'act', 'adolesc', 'ae', 'afect', 'affect', 'affectd', 'and', 'any', 'ar', 'be', 'brief', 'by', 'can', 'catch', 'caught', 'chant', 'childr', 'coronavir', 'covid-19', 'cur', 'do', 'doe', 'drug', 'ev', 'expect', 'fast', 'fin', 'for', 'from', 'get', 'go', 'grocery', 'has', 'he', 'hey', 'how', 'hug', 'i', 'if', 'in', 'infect', 'is', 'isol', 'it', 'kid', 'know', 'larg', 'maintain', 'me', 'mean', 'myself', "n't", 'no', 'numb', 'of', 'or', 'ord', 'oth', 'ourselv', 'person', 'pron', 'protect', 'quarantin', 'saf', 'safegaurd', 'self', 'shal', 'shop', 'should', 'show', 'somon', 'spread', 'spreading', 'symptom', 'tel', 'that', 'the', 'ther', 'thi', 'to', 'tre', 'vaccin', 'very', 'vir', 'want', 'way', 'we', 'wel', 'what', 'when', 'wher', 'who', 'wil', 'you']

    def bag_of_words(s, words):
        bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return numpy.array(bag)




    inp = val

    results = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    if(results[0][results_index]>0.7):
      print(tag)
      if tag=='grocery':
          return render_template("grocery.html")
      elif tag=='COVID':
          return render_template("covid.html")
      elif tag=='children':
          return render_template("child.html")
      elif tag=='isloate':
          return render_template("isolation.html")
      elif tag=='nospread':
          return render_template("nospread.html")
      elif tag=='protection':
          return render_template("protect.html")
      elif tag=='spread':
          return render_template("spread.html")
      elif tag=='symptoms':
          return render_template("sym.html")
      elif tag=='vaccine':
          return render_template("vaccine.html")
      elif tag=='virus':
          return render_template("virus.html")

    else:
      print("Sorry cant understand,please try again.")
      k="sorry"
      return render_template("sorry.html")


    #return render_template("chatbottest.html")
@app.route("/bsign")
def bsign():
    return render_template("bsign.html")
@app.route("/dlogin")
def dlogin():
    return render_template("dlogin.html")
@app.route("/hellod", methods=["POST"])
def hellod():
    email = str(request.form.get("email"))
    p = request.form.get("password")

    cur = mysql.connection.cursor()
    cur.execute('SELECT first_name FROM doctor WHERE email = %s AND password = %s', (email, p,))
    account = cur.fetchone()
    cur.close()
    global email1
    email1=email
    if account:
        res1 = ""
        for i in account:
            if i.isalpha():
                res1 = "".join([res1, i])
        account=str(res1)
        global acc
        acc=account

        cur = mysql.connection.cursor()

        cur.execute("SELECT * FROM record Order By risk DESC")
        data = cur.fetchall()
        print(type(data))
        print(data)
        return render_template("doctor.html",data=data)
    else:
        # Account doesnt exist or username/password incorrect
        return render_template("dlogin.html")


@app.route("/plogin")
def plogin():
    return render_template("plogin.html")
@app.route("/hellop", methods=["POST"])
def hellop():
    email = str(request.form.get("email"))
    p = request.form.get("password")

    cur = mysql.connection.cursor()
    cur.execute('SELECT first_name FROM patient WHERE email = %s AND password = %s', (email, p,))
    account = cur.fetchone()

    cur.close()
    global email1
    email1=email
    if account:
        res1 = ""
        for i in account:
            if i.isalpha():
                res1 = "".join([res1, i])
        account=str(res1)
        global acc
        acc=account
        return render_template("final.html", account=acc)
    else:
        # Account doesnt exist or username/password incorrect
        return render_template("plogin.html")
@app.route("/plogin")
@app.route("/dsign")
def dsign():
    return render_template("dsign.html")

@app.route("/goodd", methods=["POST"])
def goodd():
    firstname = str(request.form.get("firstName"))
    lastname = str(request.form.get("lastName"))
    email = str(request.form.get("email"))
    p = request.form.get("password")
    mo = int(request.form.get("phoneNumber"))
    cur = mysql.connection.cursor()
    cur.execute('INSERT INTO doctor(first_name,last_name,email,password,mobile) VALUES (%s,%s,%s,%s,%s);', (firstname,lastname,email,p,mo,))

    account = cur.fetchone()
    mysql.connection.commit()
    cur.close()
    global email1
    email1=email
    global acc
    acc=firstname

    return render_template("doctor.html")

@app.route("/psign")
def psign():
    return render_template("psign.html")
@app.route("/goodp", methods=["POST"])
def goodp():
    firstname = str(request.form.get("firstName"))
    lastname = str(request.form.get("lastName"))
    email = str(request.form.get("email"))
    p = request.form.get("password")

    cur = mysql.connection.cursor()
    cur.execute('INSERT INTO patient(first_name,last_name,email,password) VALUES (%s,%s,%s,%s);', (firstname,lastname,email,p,))
    account = cur.fetchone()
    mysql.connection.commit()
    cur.close()

    global email1
    email1=email
    global acc
    acc=firstname
    return render_template("final.html",account=firstname)


if __name__ == "__main__":
    app.run(debug=True)
