from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import JsonResponse
from datetime import date

from django.contrib import messages
from django.contrib.auth.models import User , auth
from .models import patient , doctor , diseaseinfo , consultation ,rating_review
from chats.models import Chat,Feedback
from sklearn.impute import SimpleImputer

from django.conf import settings
import os
import joblib as jb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from django.shortcuts import render

 #3 MODELS
# def allopathy(request):
#   # diseaselist=['Paracetamol','Ibuprofen','Aspirin','Diphenhydramine','Naproxen','Cetirizine','Ranitidine','Montelukast ',
# #  'Fluticasone','Omeprazole','Esomeprazole']


#   symptoms_list=['Headache','Fever','Cough','Sore_Throat','Fatigue','Muscle_Aches','Runny_Nose','Sneezing','Nausea','Vomiting',]

#   if request.method == 'POST':
#         # Get the input symptoms from the form
#         symptoms = request.POST.get('symptoms')
        
#         # Preprocess the input symptoms into a list of 0s and 1s
#         new_symptoms = [1 if symptom.strip() in symptoms.lower().replace(" ", "").split(',') else 0 for symptom in symptoms_list]
        
#         # Predict the probabilities of each class using the `predict_proba()` method
#         predicted_probabilities = model.predict_proba([new_symptoms])[0]
        
#         # Sort the probabilities in descending order and get the top 3 predictions
#         label_encoder = LabelEncoder()
#         top_3_predictions = label_encoder.inverse_transform(predicted_probabilities.argsort()[::-1][:3])
        
#         # Render the results in the template
#         return render(request, '/Users/Admin/Desktop/kun/Disease-Prediction-using-Django-and-machine-learning-master/templates/patient/allopathy/result.html', {'symptoms': symptoms, 'predictions': top_3_predictions})
#   else:
#         return render(request, '/Users/Admin/Desktop/kun/Disease-Prediction-using-Django-and-machine-learning-master/templates/patient/allopathy/allopathy.html')
       

def home_remedy(request):
  return render(request,'/Users/Admin/Desktop/kun/Disease-Prediction-using-Django-and-machine-learning-master/templates/patient/home_remedy/home_remedy.html')

def know(request):
  return render(request,'/Users/Admin/Desktop/kun/Disease-Prediction-using-Django-and-machine-learning-master/templates/patient/ayurvedic/know.html')
def lab_tests(request):
  return render(request,'/Users/Admin/Desktop/kun/Disease-Prediction-using-Django-and-machine-learning-master/templates/patient/checkdisease/lab_tests.html')
def allopathy(request):
  
   # Load the data into a Pandas dataframe
    data_2 = pd.read_csv("/Users/Admin/Desktop/kun/Disease-Prediction-using-Django-and-machine-learning-master/disease_prediction/model/allopathuc/Allopathic_Dataset.csv")
    
    symptoms_list = [symptom.lower().replace(' ', '') for symptom in data_2.columns.tolist() if symptom != 'Medicines']

    # Encode the categorical target variable `Medicines` into numerical labels
    label_encoder = LabelEncoder()
    data_2["Medicines"] = label_encoder.fit_transform(data_2["Medicines"])

    # Split the data into input features (symptoms) and target variable (Medicines)
    X = data_2.drop("Medicines", axis=1)
    y = data_2["Medicines"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    # y_train = imputer.fit_transform(y_train.reshape(-1, 1)).flatten()
    X_test = imputer.fit_transform(X_test)
    # y_test = imputer.fit_transform(y_test.reshape(-1, 1)).flatten()

    # Instantiate a DecisionTreeClassifier algorithm, RandomForestClassifier algorithm and KNeighborsClassifier algorithm
    model_DT = DecisionTreeClassifier(random_state=42)
    model_RF = RandomForestClassifier(random_state=42)
    model_KNN = KNeighborsClassifier(n_neighbors=3)

    # Train the model on the training set
    model_DT.fit(X_train, y_train.ravel())
    model_RF.fit(X_train, y_train.ravel())
    model_KNN.fit(X_train, y_train.ravel())

    if request.method == 'POST':
        new = request.POST.get('symptoms')
        
        if not new:
          error_message = "Please enter some symptoms."
          return render(request, '/Users/Admin/Desktop/kun/Disease-Prediction-using-Django-and-machine-learning-master/templates/patient/allopathy/allopathy.html', {'error_message': error_message})

        new = new.lower().replace(" ", "")
        new_symptoms = [1 if all(symptom.strip() in s for s in new.split(',')) else 0 for symptom in symptoms_list]

        # Predict the probabilities of each class using the `predict_proba()` method
        predicted_probabilities_DT = model_DT.predict_proba([new_symptoms])[0]
        predicted_probabilities_RF = model_RF.predict_proba([new_symptoms])[0]
        predicted_probabilities_KNN = model_KNN.predict_proba([new_symptoms])[0]

        # Sort the probabilities in descending order and get the top 3 predictions
        top_3_predictions_DT = label_encoder.inverse_transform(predicted_probabilities_DT.argsort()[::-1][:3])
        top_3_predictions_RF = label_encoder.inverse_transform(predicted_probabilities_RF.argsort()[::-1][:3])
        top_3_predictions_KNN = label_encoder.inverse_transform(predicted_probabilities_KNN.argsort()[::-1][:3])

        context = {'top_3_predictions_DT': top_3_predictions_DT,
                   'top_3_predictions_RF': top_3_predictions_RF,
                   'top_3_predictions_KNN': top_3_predictions_KNN}


        return render(request, '/Users/Admin/Desktop/kun/Disease-Prediction-using-Django-and-machine-learning-master/templates/patient/allopathy/result.html', context)

    return render(request, 'C:/Users/Admin/Desktop/kun/Disease-Prediction-using-Django-and-machine-learning-master/templates/patient/allopathy/allopathy.html')






# Define a view function for the symptoms form page



def ayurvedic(request):
  
    # Load the data into a Pandas dataframe
    data_2 = pd.read_csv("/Users/Admin/Desktop/kun/Disease-Prediction-using-Django-and-machine-learning-master/disease_prediction/model/ayurvedic/final_Ayurvedic_Dataset.csv")
    
    symptoms_list = [symptom.lower().replace(' ', '') for symptom in data_2.columns.tolist() if symptom != 'Remedy']

    # Encode the categorical target variable `Medicines` into numerical labels
    label_encoder = LabelEncoder()
    data_2["Remedy"] = label_encoder.fit_transform(data_2["Remedy"])

    # Split the data into input features (symptoms) and target variable (Medicines)
    X = data_2.drop("Remedy", axis=1)
    y = data_2["Remedy"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    # y_train = imputer.fit_transform(y_train.reshape(-1, 1)).flatten()
    X_test = imputer.fit_transform(X_test)
    # y_test = imputer.fit_transform(y_test.reshape(-1, 1)).flatten()

    # Instantiate a DecisionTreeClassifier algorithm, RandomForestClassifier algorithm and KNeighborsClassifier algorithm
    model_DT = DecisionTreeClassifier(random_state=42)
    model_RF = RandomForestClassifier(random_state=42)
    model_KNN = KNeighborsClassifier(n_neighbors=3)

    # Train the model on the training set
    model_DT.fit(X_train, y_train.ravel())
    model_RF.fit(X_train, y_train.ravel())
    model_KNN.fit(X_train, y_train.ravel())


    if request.method == 'POST':
        new = request.POST.get('symptoms')
        if not new:
            error_message = "Please enter some symptoms."
            return render(request, 'C:/Users/Admin/Desktop/kun/Disease-Prediction-using-Django-and-machine-learning-master/templates/patient/ayurvedic/ayurvedic.html', {'error_message': error_message})

        new = new.lower().replace(" ", "")
        new_symptoms = [1 if all(symptom.strip() in s for s in new.split(',')) else 0 for symptom in symptoms_list]

        # Predict the probabilities of each class using the `predict_proba()` method
        predicted_probabilities_DT = model_DT.predict_proba([new_symptoms])[0]
        predicted_probabilities_RF = model_RF.predict_proba([new_symptoms])[0]
        predicted_probabilities_KNN = model_KNN.predict_proba([new_symptoms])[0]

        # Sort the probabilities in descending order and get the top 3 predictions
        top_3_predictions_DT = label_encoder.inverse_transform(predicted_probabilities_DT.argsort()[::-1][:3])
        top_3_predictions_RF = label_encoder.inverse_transform(predicted_probabilities_RF.argsort()[::-1][:3])
        top_3_predictions_KNN = label_encoder.inverse_transform(predicted_probabilities_KNN.argsort()[::-1][:3])

        context = {'top_3_predictions_DT': top_3_predictions_DT,
                  'top_3_predictions_RF': top_3_predictions_RF,
                  'top_3_predictions_KNN': top_3_predictions_KNN}
        return render(request, '/Users/Admin/Desktop/kun/Disease-Prediction-using-Django-and-machine-learning-master/templates/patient/ayurvedic/result.html', context)

    return render(request, 'C:/Users/Admin/Desktop/kun/Disease-Prediction-using-Django-and-machine-learning-master/templates/patient/ayurvedic/ayurvedic.html')


#loading trained_model
import joblib as jb
model = jb.load('trained_model')




def home(request):

  if request.method == 'GET':
        
      if request.user.is_authenticated:
        return render(request,'homepage/index.html')

      else :
        return render(request,'homepage/index.html')



   

       


def admin_ui(request):

    if request.method == 'GET':

      if request.user.is_authenticated:

        auser = request.user
        Feedbackobj = Feedback.objects.all()

        return render(request,'admin/admin_ui/admin_ui.html' , {"auser":auser,"Feedback":Feedbackobj})

      else :
        return redirect('home')



    if request.method == 'POST':

       return render(request,'patient/patient_ui/profile.html')





def patient_ui(request):

    if request.method == 'GET':

      if request.user.is_authenticated:

        patientusername = request.session['patientusername']
        puser = User.objects.get(username=patientusername)

        return render(request,'patient/patient_ui/profile.html' , {"puser":puser})

      else :
        return redirect('home')



    if request.method == 'POST':

       return render(request,'patient/patient_ui/profile.html')

       


def pviewprofile(request, patientusername):

    if request.method == 'GET':

          puser = User.objects.get(username=patientusername)

          return render(request,'patient/view_profile/view_profile.html', {"puser":puser})





# #fin
# def allopathy(request):
#   MODEL_FILEPATH = os.path.join(settings.BASE_DIR, 'allopathic_model.joblib')

# # Load the model file
#   model = jb.load(MODEL_FILEPATH)
#   diseaselist=['Paracetamol','Ibuprofen','Aspirin','Diphenhydramine','Naproxen','Cetirizine','Ranitidine','Montelukast ',
#   'Fluticasone','Omeprazole','Esomeprazole']

 
#   symptomslist=['Headache','Fever','Cough','Sore_Throat','Fatigue','Muscle_Aches','Runny_Nose','Sneezing','Nausea','Vomiting',]

# # Define the symptoms list
#   data = pd.read_csv(os.path.join(settings.BASE_DIR, '/Users/Admin/Desktop/kun/Disease-Prediction-using-Django-and-machine-learning-master/disease_prediction/model/allopathuc/Allopathic_Dataset.csv'))
#   symptoms_list = [symptom.lower().replace(' ', '') for symptom in data.columns.tolist() if symptom != 'Medicines']
#   if request.method == 'POST':
#         # Get the input symptoms from the form
#         symptoms = request.POST.getlist('symptoms')
        
#         # Preprocess the input symptoms into a list of 0s and 1s
#         new_symptoms = [1 if symptom.strip().lower().replace(" ", "") in symptoms else 0 for symptom in symptoms_list]
        
#         # Predict the probabilities of each class using the `predict_proba()` method
#         predicted_probabilities = model.predict_proba([new_symptoms])[0]
#         label_encoder = LabelEncoder()
#         label_encoder.fit(data['Medicines'])
#         # Sort the probabilities in descending order and get the top 3 predictions
#         top_3_predictions = label_encoder.inverse_transform(predicted_probabilities.argsort()[::-1][:3])
        
#         # Render the results in the template
        
#         return render(request, 'patient/allopathy/result.html', {'symptoms': symptoms, 'predictions': top_3_predictions})
#   else:
#      return render(request, 'patient/allopathy/allopathy.html', {'symptoms_list': symptoms_list})



def checkdisease(request):

  diseaselist=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction','Peptic ulcer diseae','AIDS','Diabetes ',
  'Gastroenteritis','Bronchial Asthma','Hypertension ','Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)',
  'Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D',
  'Hepatitis E', 'Alcoholic hepatitis','Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
  'Heart attack', 'Varicose veins','Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
  'Arthritis', '(vertigo) Paroymsal  Positional Vertigo','Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo']


  symptomslist=['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
  'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination',
  'fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy',
  'patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating',
  'dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes',
  'back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
  'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
  'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
  'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
  'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
  'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
  'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
  'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
  'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
  'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
  'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
  'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
  'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
  'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
  'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
  'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
  'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
  'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
  'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
  'yellow_crust_ooze']

  alphabaticsymptomslist = sorted(symptomslist)

  


  if request.method == 'GET':
    
     return render(request,'patient/checkdisease/checkdisease.html', {"list2":alphabaticsymptomslist})




  elif request.method == 'POST':
       
      ## access you data by playing around with the request.POST object
      
      inputno = int(request.POST["noofsym"])
      print(inputno)
      if (inputno == 0 ) :
          return JsonResponse({'predicteddisease': "none",'confidencescore': 0 })
  
      else :

        psymptoms = []
        psymptoms = request.POST.getlist("symptoms[]")
       
        print(psymptoms)

      
        """      #main code start from here...
        """
      

      
        testingsymptoms = []
        #append zero in all coloumn fields...
        for x in range(0, len(symptomslist)):
          testingsymptoms.append(0)


        #update 1 where symptoms gets matched...
        for k in range(0, len(symptomslist)):

          for z in psymptoms:
              if (z == symptomslist[k]):
                  testingsymptoms[k] = 1


        inputtest = [testingsymptoms]

        print(inputtest)
      

        predicted = model.predict(inputtest)
        print("predicted disease is : ")
        print(predicted)

        y_pred_2 = model.predict_proba(inputtest)
        confidencescore=y_pred_2.max() * 100
        print(" confidence score of : = {0} ".format(confidencescore))

        confidencescore = format(confidencescore, '.0f')
        predicted_disease = predicted[0]

        

        # consult_doctor codes----------

        doctor_specialization = ["Rheumatologist","Cardiologist","ENT specialist","Orthopedist","Neurologist",
                                    "Allergist/Immunologist","Urologist","Dermatologist","Gastroenterologist"]
        

        Rheumatologist = [  'Osteoarthristis','Arthritis']
       
        Cardiologist = [ 'Heart attack','Bronchial Asthma','Hypertension ']
       
        ENT_specialist = ['(vertigo) Paroymsal  Positional Vertigo','Hypothyroidism' ]

        Orthopedist = []

        Neurologist = ['Varicose veins','Paralysis (brain hemorrhage)','Migraine','Cervical spondylosis']

        Allergist_Immunologist = ['Allergy','Pneumonia',
        'AIDS','Common Cold','Tuberculosis','Malaria','Dengue','Typhoid']

        Urologist = [ 'Urinary tract infection',
         'Dimorphic hemmorhoids(piles)']

        Dermatologist = [  'Acne','Chicken pox','Fungal infection','Psoriasis','Impetigo']

        Gastroenterologist = ['Peptic ulcer diseae', 'GERD','Chronic cholestasis','Drug Reaction','Gastroenteritis','Hepatitis E',
        'Alcoholic hepatitis','Jaundice','hepatitis A',
         'Hepatitis B', 'Hepatitis C', 'Hepatitis D','Diabetes ','Hypoglycemia']
         
        if predicted_disease in Rheumatologist :
           consultdoctor = "Rheumatologist"
           
        if predicted_disease in Cardiologist :
           consultdoctor = "Cardiologist"
           

        elif predicted_disease in ENT_specialist :
           consultdoctor = "ENT specialist"
     
        elif predicted_disease in Orthopedist :
           consultdoctor = "Orthopedist"
     
        elif predicted_disease in Neurologist :
           consultdoctor = "Neurologist"
     
        elif predicted_disease in Allergist_Immunologist :
           consultdoctor = "Allergist/Immunologist"
     
        elif predicted_disease in Urologist :
           consultdoctor = "Urologist"
     
        elif predicted_disease in Dermatologist :
           consultdoctor = "Dermatologist"
     
        elif predicted_disease in Gastroenterologist :
           consultdoctor = "Gastroenterologist"
     
        else :
           consultdoctor = "other"


        request.session['doctortype'] = consultdoctor 

        patientusername = request.session['patientusername']
        puser = User.objects.get(username=patientusername)
     

        #saving to database.....................

        patient = puser.patient
        diseasename = predicted_disease
        no_of_symp = inputno
        symptomsname = psymptoms
        confidence = confidencescore

        diseaseinfo_new = diseaseinfo(patient=patient,diseasename=diseasename,no_of_symp=no_of_symp,symptomsname=symptomsname,confidence=confidence,consultdoctor=consultdoctor)
        diseaseinfo_new.save()
        

        request.session['diseaseinfo_id'] = diseaseinfo_new.id

        print("disease record saved sucessfully.............................")

        return JsonResponse({'predicteddisease': predicted_disease ,'confidencescore':confidencescore , "consultdoctor": consultdoctor})
   


   
    



   






def dconsultation_history(request):

    if request.method == 'GET':

      doctorusername = request.session['doctorusername']
      duser = User.objects.get(username=doctorusername)
      doctor_obj = duser.doctor
        
      consultationnew = consultation.objects.filter(doctor = doctor_obj)
      
    
      return render(request,'doctor/consultation_history/consultation_history.html',{"consultation":consultationnew})



def doctor_ui(request):

    if request.method == 'GET':

      doctorid = request.session['doctorusername']
      duser = User.objects.get(username=doctorid)

    
      return render(request,'doctor/doctor_ui/profile.html',{"duser":duser})



      


def dviewprofile(request, doctorusername):

    if request.method == 'GET':

         
         duser = User.objects.get(username=doctorusername)
         r = rating_review.objects.filter(doctor=duser.doctor)
       
         return render(request,'doctor/view_profile/view_profile.html', {"duser":duser, "rate":r} )








       
def  consult_a_doctor(request):


    if request.method == 'GET':

        
        doctortype = request.session['doctortype']
        print(doctortype)
        dobj = doctor.objects.all()
        #dobj = doctor.objects.filter(specialization=doctortype)


        return render(request,'patient/consult_a_doctor/consult_a_doctor.html',{"dobj":dobj})

   


def  make_consultation(request, doctorusername):

    if request.method == 'POST':
       

        patientusername = request.session['patientusername']
        puser = User.objects.get(username=patientusername)
        patient_obj = puser.patient
        
        
        #doctorusername = request.session['doctorusername']
        duser = User.objects.get(username=doctorusername)
        doctor_obj = duser.doctor
        request.session['doctorusername'] = doctorusername


        diseaseinfo_id = request.session['diseaseinfo_id']
        diseaseinfo_obj = diseaseinfo.objects.get(id=diseaseinfo_id)

        consultation_date = date.today()
        status = "active"
        
        consultation_new = consultation( patient=patient_obj, doctor=doctor_obj, diseaseinfo=diseaseinfo_obj, consultation_date=consultation_date,status=status)
        consultation_new.save()

        request.session['consultation_id'] = consultation_new.id

        print("consultation record is saved sucessfully.............................")

         
        return redirect('consultationview',consultation_new.id)



def  consultationview(request,consultation_id):
   
    if request.method == 'GET':

   
      request.session['consultation_id'] = consultation_id
      consultation_obj = consultation.objects.get(id=consultation_id)

      return render(request,'consultation/consultation.html', {"consultation":consultation_obj })

   #  if request.method == 'POST':
   #    return render(request,'consultation/consultation.html' )





def rate_review(request,consultation_id):
   if request.method == "POST":
         
         consultation_obj = consultation.objects.get(id=consultation_id)
         patient = consultation_obj.patient
         doctor1 = consultation_obj.doctor
         rating = request.POST.get('rating')
         review = request.POST.get('review')

         rating_obj = rating_review(patient=patient,doctor=doctor1,rating=rating,review=review)
         rating_obj.save()

         rate = int(rating_obj.rating_is)
         doctor.objects.filter(pk=doctor1).update(rating=rate)
         

         return redirect('consultationview',consultation_id)





def close_consultation(request,consultation_id):
   if request.method == "POST":
         
         consultation.objects.filter(pk=consultation_id).update(status="closed")
         
         return redirect('home')






#-----------------------------chatting system ---------------------------------------------------


def post(request):
    if request.method == "POST":
        msg = request.POST.get('msgbox', None)

        consultation_id = request.session['consultation_id'] 
        consultation_obj = consultation.objects.get(id=consultation_id)

        c = Chat(consultation_id=consultation_obj,sender=request.user, message=msg)

        #msg = c.user.username+": "+msg

        if msg != '':            
            c.save()
            print("msg saved"+ msg )
            return JsonResponse({ 'msg': msg })
    else:
        return HttpResponse('Request must be POST.')



def chat_messages(request):
   if request.method == "GET":

         consultation_id = request.session['consultation_id'] 

         c = Chat.objects.filter(consultation_id=consultation_id)
         return render(request, 'consultation/chat_body.html', {'chat': c})


#-----------------------------chatting system ---------------------------------------------------


