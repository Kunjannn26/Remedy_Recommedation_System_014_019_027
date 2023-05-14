import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Load the training and testing datasets
train_data = pd.read_csv("disease_prediction\model\Training.csv")
test_data = pd.read_csv("disease_prediction\model\Testing.csv")

# Combine the datasets for preprocessing
data = pd.concat([train_data, test_data], axis=0)

# Encode the categorical target variable 'prognosis' into numerical labels
label_encoder = LabelEncoder()
data['prognosis'] = label_encoder.fit_transform(data['prognosis'])

# Split the data into input features (symptoms) and target variable (prognosis)
X = data.drop(['prognosis'], axis=1)
y = data['prognosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values using the median strategy
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Instantiate SVM, Naive Bayes, and Random Forest classifiers
model_svm = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
model_nb = GaussianNB()
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the models on the training set
model_svm.fit(X_train, y_train)
model_nb.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

# Evaluate the models on the test set
svm_score = model_svm.score(X_test, y_test)
nb_score = model_nb.score(X_test, y_test)
rf_score = model_rf.score(X_test, y_test)

# Ask for user input of symptoms
user_input = input("Enter the symptoms separated by commas: ")
new_symptoms = [1 if symptom.strip() in user_input.split(',') else 0 for symptom in X.columns]

# Predict the class using the models and the user's input
predicted_class_svm = label_encoder.inverse_transform(model_svm.predict([new_symptoms]))[0]
predicted_class_nb = label_encoder.inverse_transform(model_nb.predict([new_symptoms]))[0]
predicted_class_rf = label_encoder.inverse_transform(model_rf.predict([new_symptoms]))[0]

# Display the predicted class for the user
print(f"SVM predicted class: {predicted_class_svm}")
print(f"Naive Bayes predicted class: {predicted_class_nb}")
print(f"Random Forest predicted class: {predicted_class_rf}")

# Display the accuracy scores and confusion matrices for each model
print(f"SVM accuracy score: {svm_score}")
y_pred_svm = model_svm.predict(X_test)
cm_svm = confusion_matrix(y_test, y_pred_svm)
print(f"SVM confusion matrix:\n{cm_svm}")

print(f"Naive Bayes accuracy score: {nb_score}")
y_pred_nb = model_nb.predict(X_test)
cm_nb = confusion_matrix(y_test, y_pred_nb)
print(f"Naive Bayes confusion matrix:\n{cm_nb}")

print(f"Random Forest accuracy score: {rf_score}")
y_pred_rf = model_rf.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(f"Naive Bayes confusion matrix:\n{cm_rf}")

# Evaluate the performance of each model on the testing set
svm_accuracy = model_svm.score(X_test, y_test)
nb_accuracy = model_nb.score(X_test, y_test)
rf_accuracy = model_rf.score(X_test, y_test)

# Display the accuracy scores for each model
print(f"SVM accuracy: {svm_accuracy}")
print(f"Naive Bayes accuracy: {nb_accuracy}")
print(f"Random Forest accuracy: {rf_accuracy}")
