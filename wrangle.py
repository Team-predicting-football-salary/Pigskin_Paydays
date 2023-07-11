from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor

'How to run everything format for modeling'
'''
list_of_features = the_df.columns[2:6].to_list()
target = the_df.columns[9]
run_everything(df, list_of_features, target)
'''

def concat_dataframes():
    '''loads in all of the csv's and concats them into a single pandas dataframe'''
    df1 = pd.read_csv('2016_data')
    df2 = pd.read_csv('2017_data')
    df3 = pd.read_csv('2018_data')
    df4 = pd.read_csv('2019_data')
    df5 = pd.read_csv('2020_data')
    df6 = pd.read_csv('2021_data')
    df7 = pd.read_csv('2022_data')

    return pd.concat([df1,df2,df3,df4,df5,df6,df7])


def run_everything(df, list_of_features, target):
    '''Runs the modeling'''
    train, validate, test = split_data(df, target)
    X_train, X_validate, X_test, y_train, y_validate, y_test = get_X_train_val_test(train, validate, test, list_of_features, target)
    metrics_train_df, metrics_validate_df, metrics_test_df = get_model_numbers(X_train, X_validate, X_test, y_train, y_validate, y_test)
    return metrics_train_df, metrics_validate_df, metrics_test_df


def get_X_train_val_test(train,validate, test, list_of_features,target):
    '''
    geting the X's and y's and returns them
    '''
    X_train = train[list_of_features]
    X_validate = validate[list_of_features]
    X_test = test[list_of_features]
    
    y_train = train[target]
    y_validate = validate[target]
    y_test = test[target]

    return X_train, X_validate, X_test, y_train, y_validate, y_test


def split_data(df, target):
    '''
    Takes in a dataframe and returns train, validate, test subset dataframes
    '''
    
    
    train, test = train_test_split(df,
                                   test_size=.2, 
                                   random_state=123, 
                                   
                                   )
    train, validate = train_test_split(train, 
                                       test_size=.25, 
                                       random_state=123, 
                                       
                                       )
    
    return train, validate, test


def metrics_reg(y, yhat):
    '''
    send in y_true, y_pred and returns rmse, r2
    '''
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2


def get_model_numbers(X_train, X_validate, X_test, y_train, y_validate, y_test):
    '''
    This function takes the data and runs it through various models and returns the
    results in pandas dataframes for train, test and validate data
    '''
    baseline = y_train.mean()
    baseline_array = np.repeat(baseline, len(X_train))
    rmse, r2 = metrics_reg(y_train, baseline_array)

    metrics_train_df = pd.DataFrame(data=[
    {
        'model_train':'baseline',
        'rmse':rmse,
        'r2':r2
    }
    ])
    metrics_validate_df = pd.DataFrame(data=[
    {
        'model_validate':'baseline',
        'rmse':rmse,
        'r2':r2
    }
    ])
    metrics_test_df = pd.DataFrame(data=[
    {
        'model_validate':'baseline',
        'rmse':rmse,
        'r2':r2
    }
    ])


    Linear_regression1 = LinearRegression()
    Linear_regression1.fit(X_train,y_train)
    predict_linear = Linear_regression1.predict(X_train)
    rmse, r2 = metrics_reg(y_train, predict_linear)
    metrics_train_df.loc[1] = ['ordinary least squared(OLS)', rmse, r2]

    predict_linear = Linear_regression1.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, predict_linear)
    metrics_validate_df.loc[1] = ['ordinary least squared(OLS)', rmse, r2]
    
    predict_linear = Linear_regression1.predict(X_test)
    rmse, r2 = metrics_reg(y_test, predict_linear)
    metrics_test_df.loc[1] = ['ordinary least squared(OLS)', rmse, r2]


    lars = LassoLars()
    lars.fit(X_train, y_train)
    pred_lars = lars.predict(X_train)
    rmse, r2 = metrics_reg(y_train, pred_lars)
    metrics_train_df.loc[2] = ['lasso lars(lars)', rmse, r2]

    pred_lars = lars.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, pred_lars)
    metrics_validate_df.loc[2] = ['lasso lars(lars)', rmse, r2]
    
    pred_lars = lars.predict(X_test)
    rmse, r2 = metrics_reg(y_test, pred_lars)
    metrics_test_df.loc[2] = ['lasso lars(lars)', rmse, r2]


    pf = PolynomialFeatures(degree=2)
    X_train_degree2 = pf.fit_transform(X_train)

    pr = LinearRegression()
    pr.fit(X_train_degree2, y_train)
    pred_pr = pr.predict(X_train_degree2)
    rmse, r2 = metrics_reg(y_train, pred_pr)
    metrics_train_df.loc[3] = ['Polynomial Regression(poly2)', rmse, r2]

    X_validate_degree2 = pf.transform(X_validate)
    pred_pr = pr.predict(X_validate_degree2)
    rmse, r2 = metrics_reg(y_validate, pred_pr)
    metrics_validate_df.loc[3] = ['Polynomial Regression(poly2)', rmse, r2]

    X_test_degree2 = pf.transform(X_test)
    pred_pr = pr.predict(X_test_degree2)
    rmse, r2 = metrics_reg(y_test, pred_pr)
    metrics_test_df.loc[3] = ['Polynomial Regression(poly2)', round(rmse,2), r2]

    
    glm = TweedieRegressor(power=2, alpha=0)
    glm.fit(X_train, y_train)
    
    pred_glm = glm.predict(X_train)
    rmse, r2 = metrics_reg(y_train, pred_glm)
    metrics_train_df.loc[4] = ['Generalized Linear Model (GLM)', rmse, r2]

    pred_glm = glm.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, pred_glm)
    metrics_validate_df.loc[4] = ['Generalized Linear Model (GLM)', rmse, r2]
    
    pred_glm = glm.predict(X_test)
    rmse, r2 = metrics_reg(y_test, pred_glm)
    metrics_test_df.loc[4] = ['Generalized Linear Model (GLM)', rmse, r2]


    metrics_train_df.rmse = metrics_train_df.rmse.astype(int)
    metrics_validate_df.rmse = metrics_validate_df.rmse.astype(int)
    metrics_test_df.rmse = metrics_test_df.rmse.astype(int)
    print()
    metrics_train_df.r2 = (metrics_train_df.r2 * 100).astype(int)
    metrics_validate_df.r2 = (metrics_validate_df.r2 * 100).astype(int)
    metrics_test_df.r2 = (metrics_test_df.r2 * 100).astype(int)

    return metrics_train_df, metrics_validate_df, metrics_test_df


def univariate_visual(df):
    '''
    creates histplots for all of my columns
    '''
    plt.figure(figsize=(25,15))
    plt.xticks(rotation = 45)
    for i, col in enumerate(df):

        plt.title(col)
        sns.histplot(df[col])
        plt.xticks(rotation=45)
        plt.show()
        
    plt.show()