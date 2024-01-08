import pandas as pds
import matplotlib.pyplot as mat
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Load dataset
columns = ["url", "type"]

file = "C:/Users/acer/Desktop/PhishingURLDataset.xlsx"
data = pds.read_excel(file, names=columns)

print("=============== DIMENSION ===============")
print(data.shape) 

print("\n============ STATISTIC DATA =============")
print(data.describe()) 

print()
print("\n================ SUMMARY ================")
print("URL Type Description")
print("\nBenign URLs:\nThese are safe to browse URLs.")
print("\nMalware URLs:\nThese type of URLs inject malware into the victim's system once he/she visit such URLs.")
print("\nDefacement URLs:\nThese type of URLs are generally created by hackers with the intention of breaking into a web server")
print("and replacing the hosted website with one of their own, using techniques such as code injection, cross-site scripting, etc.")
print("\nPhishing URLs:\nThese type of URLs are generally created by hackers that will try to steal sensitive personal")
print("or financial information such as login credentials, credit card numbers, internet banking details, etc.")

print("\nDataset Summary\n")
print(data.groupby("type").size())

# Visualization
# Count of links per type =================================
mat.figure(figsize=(8, 5))
sns.countplot(x='type', data=data)
mat.title("Number of URLs per type")
mat.show()

type_counts = data['type'].value_counts()

# Pie chart of links per type (which gives percentage of per type) =================================
mat.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
mat.axis('equal')
mat.title('Distribution of URL Types')
mat.show()

# Split data
df = pds.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(df['url'], df['type'], test_size=0.2, random_state=42)

print("\n================= X-TEST ================")
print(X_test)

print("\n================= Y-TEST ================")
print(y_test)

print("\n=============== ALGORITHM ===============")

# Model Training 

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Linear Support Vector Machine (SVM) Model
svm_model = LinearSVC(dual="auto")
svm_model.fit(X_train_vectorized, y_train)

# Predictions
predictions = svm_model.predict(X_test_vectorized)

accuracy = f"{accuracy_score(y_test, predictions) * 100:.2f}%" 
recall = f"{recall_score(y_test, predictions, average='weighted') * 100:.2f}%" 
precision = f"{precision_score(y_test, predictions, average='weighted') * 100:.2f}%" 
f1 = f"{f1_score(y_test, predictions, average='weighted') * 100:.2f}%" 

# Model Evaluation 
print("\nAccuracy:", accuracy)
print("Recall Score:", recall)
print("Precision Score:", precision)
print("F1 Score:", f1)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Validation/Testing in the new data
test_file = "C:/Users/acer/Desktop/TestData.xlsx"
test_data = pds.read_excel(test_file, usecols=["url"])
X_new_test = test_data["url"]

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train) # convert text values into matrix tokens
new_url_vectorized = vectorizer.transform(X_new_test) # store matrix tokens in the X_test_vectorized variable

model = LinearSVC(dual="auto")
model.fit(X_train_vectorized, y_train)
prediction = model.predict(new_url_vectorized)   # pass the stored matrix tokens in to the model

print("\n============== PREDICTION ===============")
print(f"Number of Records: {len(prediction.tolist())}\n")

# list all validation data's URL and their predicted type
for count, (pred, url) in enumerate(zip(prediction.tolist(), test_data["url"]), 1):
    print(f"#{count}")
    print(f"URL:        {url}")
    print(f"Prediction: {pred}\n")