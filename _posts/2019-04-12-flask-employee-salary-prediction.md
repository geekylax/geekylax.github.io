<!-- ---
layout: post
title: Flask API
subtitle: Flask deployment for employee salary prediction 

tags: [python , flask , heroku , salaryprediction]
--- -->
<!-- bigimg: /img/webapp.png
 -->
---
layout: post
title: Flask API
subtitle: Flask deployment for employee salary prediction 
bigimg: /img/webapp.png
tags: [python,flask,heroku,salaryprediction]
---

## From creating a machine learning model to deploying it on the web — a succinct guide


I just  build a prediction model on historic data using different machine learning algorithms and classifiers, plot the results and calculate the accuracy of the model on the testing data. Now what? To put it to use in order to predict the new data we have to deploy it over the internet so that the outside world can use it. In this article I talk about how I trained a machine learning model, created a web application on it using Flask and deployed it using Heroku.

**Training Decision Tree**


[![Decision tree]({{ site.url }}/img/decisiontree.gif)]({{ site.url }}/img/decisiontree.gif){: .center-image }


Decision Tree is a well known supervised machine learning algorithm because its ease to use, resilient and flexible. I have implemented the algorithm on Adult dataset from UCI machine learning repository [here](https://archive.ics.uci.edu/ml/datasets/adult). In one of my previous articles I have in-depth explained the dataset and compared different classifiers I trained on it. Please feel free to check it out [here](https://towardsdatascience.com/comparative-study-of-classifiers-in-predicting-the-income-range-of-a-person-from-a-census-data-96ce60ee5a10).


**Preprocessing the dataset**

It consists of 14 attributes and a class label telling whether the income of the individual is less than or more than 50K a year. These attributes range from the age of the person, the working class label to relationship status and the race the person belongs to. The information about all the attributes can be found here.

At first we find and remove any missing values from the data. I have replaced the missing values with the mode value in that column. There are many other ways to replace missing values but for this type of dataset it seemed most optimal.

To fit the data into prediction model, we need convert categorical values to numerical ones. Before that, we will evaluate if any transformation on categorical columns are necessary. Discretisation is a common way to make categorical data more tidy and meaningful. I applied discretisation on column marital_status where they are narrowed down to only to values married or not married. Later I apply label encoder in the remaining data columns. Also there are two redundant columns {‘education’, ‘educational-num’}, therefore I have removed one of them.

~~~
# Load dataset
url = "Dataset/adult.csv"
df = pandas.read_csv(url)

# filling missing values
col_names = df.columns
for c in col_names:
    df[c] = df[c].replace("?", numpy.NaN)

df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

#discretisation
df.replace(['Divorced', 'Married-AF-spouse', 
              'Married-civ-spouse', 'Married-spouse-absent', 
              'Never-married','Separated','Widowed'],
             ['divorced','married','married','married',
              'not married','not married','not married'], inplace = True)

#label Encoder
category_col =['workclass', 'race', 'education','marital-status', 'occupation',
               'relationship', 'gender', 'native-country', 'income'] 
labelEncoder = preprocessing.LabelEncoder()

# creating a map of all the numerical values of each categorical labels.
mapping_dict={}
for col in category_col:
    df[col] = labelEncoder.fit_transform(df[col])
    le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col]=le_name_mapping
print(mapping_dict)

#droping redundant columns
df=df.drop(['fnlwgt','educational-num'], axis=1)
~~~


Now this is what the data set looks like:

[![data]({{ site.url }}/img/data.png)]({{ site.url }}/img/data.png)


**Fitting the model**

After preprocessing the data for each classifier depending on how the data is fed, we then slice the data separating the labels with the attributes. Now, we split the dataset into two halves, one for training and on for testing. This is achieved using train_test_split() function of sklearn.

~~~
X = df.values[:, 0:12]
Y = df.values[:,12]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
dt_clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=5, min_samples_leaf=5)
dt_clf_gini.fit(X_train, y_train)
y_pred_gini = dt_clf_gini.predict(X_test)

print ("Desicion Tree using Gini Index\nAccuracy is ", accuracy_score(y_test,y_pred_gini)*100 )

#creating and training a model
#serializing our model to a file called model.pkl
import pickle
pickle.dump(dt_clf_gini, open(".../model.pkl","wb"))

~~~


With this we achieve an accuracy of 84% approximately. Now in order to use this model with new unknown data we need to save the model so that we can predict the values later. For this we make use of pickle in python which is a powerful algorithm for serializing and de-serializing a Python object structure.



## Creating a Simple Web Application using Flask
 
[![flask]({{ site.url }}/img/flask.png)]({{ site.url }}/img/flask.png)


**HTML Form**
For predicting the income from various attributes we first need to collect the data(new attribute values) and then use the decision tree model we build above to predict whether the income is more than 50K or less. Therefore, in order to collect the data we create html form which would contain all the different options to select from each attribute. Here, I have created a simple form using html only. You can review the code . If you want to make the form more interactive you can do so as well.


**Example of the form**



**Important:**in-order to predict the data correctly the corresponding values of each label should match with the value of each input selected.
{: .box-note}

For example — In the attribute Relationship there are 6 categorical values. These are converted to numerical like this 

{‘Husband’: 0, ‘Not-in-family’: 1, ‘Other-relative’: 2, ‘Own-child’: 3, ‘Unmarried’: 4, ‘Wife’: 5}.

Therefore we need to put the same values to the html form.
~~~
<label for="relation">Relationship</label>
    <select id="relation" name="relation">
      <option value="0">Husband</option>
      <option value="1">Not-in-family</option>
      <option value="2">Other-relative</option>
      <option value="3">Own-child</option>
      <option value="4">Unmarried</option>
      <option value="5">Wife</option>
    </select>
~~~

In the gist preprocessing.py above I have created a dictionary mapping_dict which stores the numerical values of all the categorical labels in the form of key and value. This would help in creating the html form.

Till now we have created the html form and now to host the static pages we need to use flask.


##Flask script

Before starting with the coding part, we need to download flask and some other libraries. Here, we make use of virtual environment, where all the libraries are managed and makes both the development and deployment job easier.
~~~
mkdir income-prediction
cd income-prediction
virtualenv env
~~~
After the virtual environment is created we activate it.

~~~
set venv/bin/activate.bat
~~~

Now let’s install Flask.

~~~
pip install flask
~~~

Lets create folder templates. In your application, you will use templates to render HTML which will display in the user’s browser. This folder contains our html form file index.html.

mkdir templates

Create script.py file in the project folder and copy the following code.

~~~
#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')
~~~

Here we import the libraries, then using app=Flask(__name__) we create an instance of flask. @app.route('/') is used to tell flask what url should trigger the function index() and in the function index we use render_template('index.html') to display the script index.html in the browser.

Let’s run the application

~~~
export FLASK_APP=script.py
run flask
~~~

This should run the application and launch a simple server. Open http://127.0.0.1:5000/ to see the html form.

Predicting the income value
When someone submits the form, the webpage should display the predicted value of income. For this, we require the model file(model.pkl) we created before, in the same project folder.

~~~
#prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,12)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        
        if int(result)==1:
            prediction='Income more than 50K'
        else:
            prediction='Income less that 50K'
            
        return render_template("result.html",prediction=prediction)
~~~


Here after the form is submitted, the form values are stored in variable to_predict_list in the form of dictionary. We convert it into a list of the dictionary’s values and pass it as an argument to ValuePredictor() function. In this function, we load the model.pkl file and predict the new values and return the result.

This result/prediction(Income more than or less than 50k) is then passed as an argument to the template engine with the html page to be displayed.

Create the following result.html file and add it to templates folder.


Run the application again and it should predict the income after submitting the form.

We have successfully created the Webapp. Now it’s time to use heroku to deploy it.
This is how our project layout looks like 

~~~
/income-prediction
   ├── templates
   │   └── index.html
   ├── venv/
   ├── model.pkl
   └── setup.py
~~~
Deploying Flask app using Heroku
Related image
Heroku is a platform as a service (PaaS) that enables developers to build, run, and operate applications entirely in the cloud. In this project we deploy using heroku git.

For this we need to install git as well as heroku CLI onto our system. Please refer to these links — [Git],[Heroku]. Now, visit Heroku and create an account.

Let’s get started —

Step 1:
At first we need to download gunicorn to our virtual environment venv. We can use pip to download it.

~~~
pip install gunicorn
~~~
Gunicorn handles requests and takes care of complicated things like threading very easily and the server that we use to run Flask locally when we’re developing our application isn’t good at handling real requests, therefore we use gunicorn.

Step 2:
In our local machine, we have installed a lot of libraries and other important files like flask, gunicorn, sklearn etc. We need to tell heroku that our project requires all these libraries to successfully run the application. This is done by creating a requirements.txt file.

~~~
pip freeze > requirements.txt

~~~
Step 3:
Procfile is a text file in the root directory of your application, to explicitly declare what command should be executed to start your app. This is an essential requirement for heroku.


This file tells heroku we want to use the web process with the command gunicorn and the app name.

Step 4:
In our project folder we have a lot of hidden or unnecessary files which we do not want to deploy to heroku. For example venv, instance folders or .cache files. In order to not include them we create a .gitignore file.

~~~
venv/

*.pyc
__pycache__/

instance/

.pytest_cache/
.coverage
htmlcov/

dist/
build/
*.egg-info/

.DS_Store
~~~


This is how our project layout looks like —
~~~
/income-prediction
   ├── templates
   │   └── index.html
   ├── venv/
   ├── Procfile
   ├── requirements.txt
   ├── .gitignore
   ├── model.pkl
   └── setup.py
~~~   

**Now our project is ready to be pushed to heroku**.

Step 5:

Open terminal and execute the following commands —

heroku login
This is will ask for your heroku credentials. Now, we need to create a heroku app.

heroku create
This will create a heroku app with a system generated url. We can manually change the url later using a set of commands.

~~~
git init
git add .
git commit -m 'initial commit'
~~~

This initialises the repo, adds all the codes and commits it with a message.

~~~
git push heroku master
heroku open
~~~
This will push the entire app on heroku and open the url in the browser.

Done 

It was that simple. We created a machine learning model, trained it, created a web application to predict new data using the model and deployed it on the internet using heroku. And did all of it in python!

