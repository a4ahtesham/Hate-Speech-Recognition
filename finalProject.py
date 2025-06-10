#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


import numpy as np


# In[ ]:


import nltk


# In[ ]:


import re


# In[ ]:


import string


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# In[16]:


import nltk
nltk.download('stopwords')


# In[17]:


# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


# In[18]:


stopword=set(stopwords.words('english'))
stemmer = nltk. SnowballStemmer("english")


# In[19]:


#Previewing the CSV File Data
data = pd.read_csv("C:\\Users\\SAIF ALI KHAN\\Downloads\\labeled_data.csv")
print(data. head())


# In[20]:


data["labels"] = data["class"]. map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
data = data[["tweet", "labels"]]
print(data. head(10))


# In[21]:


def clean (text):
 text = str (text). lower()
 text = re. sub('[.?]', '', text) 
 text = re. sub('https?://\S+|www.\S+', '', text)
 text = re. sub('<.?>+', '', text)
 text = re. sub('[%s]' % re. escape(string. punctuation), '', text)
 text = re. sub('\n', '', text)
 text = re. sub('\w\d\w', '', text)
 text = [word for word in text.split(' ') if word not in stopword]
 text=" ". join(text)
 text = [stemmer. stem(word) for word in text. split(' ')]
 text=" ". join(text)
 return text
data["tweet"] = data["tweet"]. apply(clean)


# In[22]:


x = np. array(data["tweet"])
y = np. array(data["labels"])
cv = CountVectorizer()
X = cv. fit_transform(x)
#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[23]:


# Classifiers to try
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "MLP Classifier": MLPClassifier(max_iter=300)
}


# In[24]:


# Evaluate all models
best_model = None
best_accuracy = 0
print("Model Accuracy Scores:\n")


# In[ ]:


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.4f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model


# In[ ]:


inp = input("\nEnter the Speech to be tested: ")
inp_cleaned = clean(inp)
inp_vectorized = cv.transform([inp_cleaned]).toarray()
prediction = best_model.predict(inp_vectorized)
print("Prediction:", prediction[0])


# In[ ]:


import tkinter as tk
from tkinter import messagebox


# In[ ]:


# GUI setup
app = tk.Tk()
app.title("Hate Speech Detection")
app.geometry("500x300")
app.resizable(False, False)

tk.Label(app, text="Enter Text:", font=("Arial", 14)).pack(pady=10)
text_entry = tk.Text(app, height=5, width=50, font=("Arial", 12))
text_entry.pack()

tk.Button(app, text="Detect", command=predict_speech, font=("Arial", 12), bg="blue", fg="white").pack(pady=10)

result_label = tk.Label(app, text="", font=("Arial", 14))
result_label.pack(pady=10)

app.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




