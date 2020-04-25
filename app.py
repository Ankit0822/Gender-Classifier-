from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle

app=Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    df=pd.read_csv('gender.csv')
    df_X=df.name
    df_Y=df.sex

    corpus=df_X
    cv=CountVectorizer()
    X=cv.fit_transform(corpus)
    X_train,X_test,Y_train,Y_test=train_test_split(X,df_Y,test_size=0.3,random_state=42)
    clf=MultinomialNB()
    clf.fit(X_train,Y_train)
    clf.score(X_test,Y_test)
    file=open('model.pkl','wb')
    pickle.dump(clf,file)
    file.close()

    if request.method=='POST':
        namequery=request.form['namequery']
        data=[namequery]
        vect=cv.transform(data).toarray()
        my_prediction=clf.predict(vect)
    return render_template('result.html',prediction=my_prediction,name=namequery.upper())




if __name__ == '__main__':
    app.run(debug=True)