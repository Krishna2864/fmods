
# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse


def home(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        username = float(username)
        username = int(fun(username))

        return render(request, 'anuman/result.html', {'username': username})

    return render(request, 'anuman/home.html', {})



def fun(x):
    # Simple Linear Regression

    # Importing the libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # Importing the dataset
    dataset = pd.read_csv('b/Salary_Data.csv')
    X = dataset.iloc[:, :-1].values  # 0
    y = dataset.iloc[:, 1].values

    # Splitting the dataset into the Training set and Test set
    # from sklearn.cross_validation import train_test_split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

    # Feature Scaling
    """from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

    # Fitting Simple Linear Regression to the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    return regressor.predict([[x]])

