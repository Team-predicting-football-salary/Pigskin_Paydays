from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor
from sklearn.feature_selection import SelectKBest, RFE, f_regression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


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


def split_data(df):
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
    predict_linear_train = Linear_regression1.predict(X_train)
    rmse, r2 = metrics_reg(y_train, predict_linear_train)
    metrics_train_df.loc[1] = ['ordinary least squared(OLS)', rmse, r2]
    feature_weights = Linear_regression1.coef_

    predict_linear_validate = Linear_regression1.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, predict_linear_validate)
    metrics_validate_df.loc[1] = ['ordinary least squared(OLS)', rmse, r2]
    
    predict_linear_test = Linear_regression1.predict(X_test)
    rmse, r2 = metrics_reg(y_test, predict_linear_test)
    metrics_test_df.loc[1] = ['ordinary least squared(OLS)', rmse, r2]

    lars = LassoLars()
    lars.fit(X_train, y_train)
    pred_lars = lars.predict(X_train)
    rmse, r2 = metrics_reg(y_train, pred_lars)
    metrics_train_df.loc[2] = ['lasso lars(lars)', rmse, r2]

    pred_lars = lars.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, pred_lars)
    metrics_validate_df.loc[2] = ['lasso lars(lars)', rmse, r2]


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
    
    glm = TweedieRegressor(power=2, alpha=0)
    glm.fit(X_train, y_train)
    
    pred_glm = glm.predict(X_train)
    rmse, r2 = metrics_reg(y_train, pred_glm)
    metrics_train_df.loc[4] = ['Generalized Linear Model (GLM)', rmse, r2]

    pred_glm = glm.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, pred_glm)
    metrics_validate_df.loc[4] = ['Generalized Linear Model (GLM)', rmse, r2]

    rfr = RandomForestRegressor(n_estimators=100, random_state=42)  
    rfr.fit(X_train, y_train)  
    pred_rfr = rfr.predict(X_train)
    rmse, r2 = metrics_reg(y_train, pred_rfr)
    metrics_train_df.loc[5] = ['Random Forest Regressor', rmse, r2]

    pred_rfr = rfr.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, pred_rfr)
    metrics_validate_df.loc[5] = ['Random Forest Regressor', rmse, r2]

    return metrics_train_df, metrics_validate_df, metrics_test_df, predict_linear_train, feature_weights, predict_linear_test


def univariate_visual(df):
    '''
    creates histplots for all of my columns
    '''
    plt.figure(figsize=(25,15))
    plt.xticks(rotation = 45)
    count = 0
    for i, col in enumerate(df):
        if count < 6:
            plt.title(col)
            sns.histplot(df[col])
            plt.xticks(rotation=45)
            plt.show()
        count +=1
    plt.show()


def correlation_charts(train,columns_list, target):
    '''
    Creates and shows visuals for Correlation tests 
    '''
    plt.figure(figsize=(14,3))
    # plt.suptitle('Bivariate Exploration: The Strongest Correlators of target variable')
    for i, col in enumerate(train[columns_list]):
        if col != target:

            sns.regplot(data = train, x = col, y = target, scatter_kws={'alpha': 0.1}, line_kws={'color': 'red'})

            plt.show()


def correlation_tests(train, columns_list, target):
    '''
    Runs a correlation test on dataframe features vs target variable
    '''
    corr_df = pd.DataFrame({'feature': [],
                        'r': [],
                       'p': []})
    for i, col in enumerate(train[columns_list]):
        r, p = stats.pearsonr(train[col], train[target])
        corr_df.loc[i] = [col, r, p]
    to_return = corr_df.sort_values(by='p', ascending=False)

    return to_return


def scale_data(train,
               validate,
               test,
               cols):
    '''Takes in train, validate, and test set, and outputs scaled versions of the columns that were sent in as dataframes'''
    #Make copies for scaling
    train_scaled = train.copy() #Ah, making a copy of the df and then overwriting the data in .transform() to remove warning message
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #Initiate scaler, using Min max scaler
    scaler = MinMaxScaler()
    #Fit to train only
    scaler.fit(train[cols])
    #Creates scaled dataframes of train, validate, and test. This will still preserve columns that were not sent in initially.
    train_scaled[cols] = scaler.transform(train[cols])
    validate_scaled[cols] = scaler.transform(validate[cols])
    test_scaled[cols] = scaler.transform(test[cols])

    return train_scaled, validate_scaled, test_scaled


def moving_forward(p_val):
    '''
    This function returns whether or not a p value is less than alpha
    '''
    if p_val < .05:
        return 'Yes'
    else:
        return 'No'
    
    
def positive_negative(r_value):
    '''
    This runction returns whether or not there is positive or negative correlation
    '''
    if r_value < 0:
        return 'Negative'
    elif r_value > 0:
        return 'Positive'
    else:
        return 'Neutral'
    

def get_explore_data(columns_list, corr_test):
    '''
    Creates the explore DataFrame to show exploratory analysis
    '''
    explore = pd.DataFrame(columns_list)
    explore.columns = ['Features']
    explore['Correlation'] = corr_test['r'].apply(positive_negative)
    explore['Moving Forward'] = corr_test['p'].apply(moving_forward)
    return explore


def best_features(X_train, y_train):
    '''
    Uses Kbest object to find the best features for our model
    '''
    kbest = SelectKBest(f_regression, k=2)
    kbest.fit(X_train, y_train) 
    kbest_results = pd.DataFrame(
                    dict(p=kbest.pvalues_, f=kbest.scores_),
                    index = X_train.columns)
    return kbest_results.sort_values(by=['p', 'f'])


def rfe(X_train, y_train, the_k):
    '''
    Finds best features for LinearRegression, LassoLars, and GLM and ranks them
    '''
    Linear_regression = LinearRegression()
    lars = LassoLars()
    glm = TweedieRegressor(power=2, alpha=0)
    
    model_names = ['Linear_regression','LassoLars','GLM']
    models = [Linear_regression, lars,glm]
    master_df = pd.DataFrame()
    
    for j in range(len(models)):
        for i in range(1):
            rfe = RFE(models[j], n_features_to_select=the_k)
            rfe.fit(X_train, y_train)
            the_df = pd.DataFrame(
            {'rfe_ranking':rfe.ranking_, 'Model':model_names[j]},
            index=X_train.columns)
            master_df = pd.concat( [master_df, the_df.sort_values(by='rfe_ranking')])
            
    return master_df


def run_fold(df, columns_list, target):
    '''
    Scales the data then
    runs the train test method to split the data 
    then runs it through various models with various hyperparameters
    '''
    scaler = MinMaxScaler()
    scaler.fit(df[columns_list])
    df[columns_list] = scaler.transform(df[columns_list])

    X = df[columns_list]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=123)
    parameters_lars = {
        'alpha':range(1,21),
        'fit_intercept':[True, False],
        'verbose': [True, False]
    }

    parameters_linear = {'n_jobs':range(1,21)
                    }

    parameters_glm = {
        'link':['auto', 'identity', 'log'],
        'alpha':range(1,21),
    }
    
    lars = LassoLars(random_state =123)
    Linear_regression1 = LinearRegression()
    glm = TweedieRegressor(power=2, alpha=0)


    the_parameters = [parameters_lars, parameters_linear, parameters_glm]

    best_parameters = []
    models = ['lars','Linear_regression1','glm']
    master_df = pd.DataFrame()
    for number, tree in enumerate([lars, Linear_regression1, glm]):
        grid = GridSearchCV(tree, the_parameters[number], cv=5, scoring = 'neg_root_mean_squared_error')
        grid.fit(X_train, y_train)
        
        for p, score in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score']):
            p['score'] = abs(score)
            p['model'] = models[number]
        new_df = pd.DataFrame(pd.DataFrame(grid.cv_results_['params']).sort_values('score', ascending=True))

        best_parameters.append(grid.best_estimator_)
        master_df = pd.concat([master_df, new_df])

    baseline = y_train.mean()
    baseline_array = np.repeat(baseline, len(X_train))
    rmse, r2 = metrics_reg(y_train, baseline_array)

    metrics_test_df = pd.DataFrame(data=[
    {
        'model_validate':'baseline',
        'rmse':rmse,
        'r2':r2
    }
    ])

    pred_lars = best_parameters[0].predict(X_test)
    rmse, r2 = metrics_reg(y_test, pred_lars)
    metrics_test_df.loc[1] = ['lasso lars(lars)', rmse, r2]

    return master_df,metrics_test_df


def univariate_findings():
    '''
    Creates a pandas dataframe that shows various findings during the univariate analysis
    '''
    the_dict = {'games_played':'left_skewed', 'comp': 'non-symmetric bimodal', 'att':'non-symmetric bimodal','comp_pct':'normally', 'yds':'non-symmetric bimodal','avg_yds_per_att':'normally','td':'right-skewed', 'int':'right-skewed', 'pass_rating':'normally', 'rush_att':'right-skewed', 'rush_yds': 'right-skewed', 'rush_avg':'normally', 'rush_td':'right-skewed', 'age':'right-skewed', 'td_perc':'normally', 'int_perc': 'non-symmetric bimodal','fir_dn_throws':'non-symmetric bimodal', 'Lng_comp':'normally', 'yds_per_comp':'normally', 'yds_per_gm':'normally', 'QBR':'normally', 'sk':'normally', '4QC':'right-skewed', 'GWD': 'right-skewed'}
    key_list=[]
    value_list = []

    for key, value in the_dict.items():
        key_list.append(key)
        value_list.append(value)
    the_df = pd.DataFrame()
    the_df['Feature'] = key_list
    the_df['Distribution'] = value_list
    return the_df


def get_target_and_columns(df, train):
    '''
    Preps the data and runs a correlation test
    '''
    columns_list = df.select_dtypes(exclude=['object']).columns.to_list()
    columns_list.remove('percent_of_cap')
    target = 'percent_of_cap'
    corr_test = correlation_tests(train, columns_list , target).reset_index().drop(columns = 'index')
    corr_test['Moving Forward'] = corr_test['p'].apply(moving_forward)
    return columns_list, target, corr_test


def new_visual_univariate_findings(df):
    '''
    This function displays all of our histplots during the univariate analysis
    '''
    count = 0
    for col in df.select_dtypes(include=['object']).columns[:1]:                   

        num_cols = len(df.select_dtypes(exclude=['object']).columns[:6])
        num_rows, num_cols_subplot = divmod(num_cols, 3)
        if num_cols_subplot > 0:
            num_rows += 1

        fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
        if count < 1:
            for i, col in enumerate(df.select_dtypes(exclude=['object']).columns[:6]):
                
                row_idx, col_idx = divmod(i, 3)
                sns.histplot(df[col], ax=axes[row_idx, col_idx])
                axes[row_idx, col_idx].set_title(f'Histogram of {col}')
            
            

            plt.tight_layout()
            plt.show()


def new_visual_multivariate_findings(df, target):
    '''
    This function displays all the regplots for our bivariate analysis
    '''
    for col in df.select_dtypes(include=['object']).columns[:1]:                   

        num_cols = len(df.select_dtypes(exclude=['object']).columns[:6])
        num_rows, num_cols_subplot = divmod(num_cols, 3)
        if num_cols_subplot > 0:
            num_rows += 1

        fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))

        for i, col in enumerate(df.select_dtypes(exclude=['object']).columns[:6]):
            row_idx, col_idx = divmod(i, 3)
            sns.regplot(data = df, x = col, y = target, scatter_kws={'alpha': 0.1}, line_kws={'color': 'red'},ax=axes[row_idx, col_idx])
            
            axes[row_idx, col_idx].set_title(f'Scatterplot of {col} and {target}')

        plt.tight_layout()
        plt.show()

#====================== NATURAL LANGUAGE PROCESSING ================================

#Standard imports

import pandas as pd
import numpy as np

#Imports for webscraping

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager

import time as t
import os

import requests
from bs4 import BeautifulSoup

#NLP tokens for processing

import unicodedata
import re
import json
import nltk

import spacy

from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

#Sentiment Analysis

import nltk.sentiment

#Visualizations

import matplotlib.pyplot as plt
import seaborn as sns

def acquire_commentary():
    test_df_2 = pd.read_csv('player_commentaries_health.csv', index_col=0)
    df = pd.read_csv('player_commentaries.csv', index_col=0)

    combined_df = df.merge(test_df_2, left_on = 'player_name', right_on = 'player_name')

    combined_df['player_commentary_z'] = combined_df['player_commentary_x'] + combined_df['player_commentary_y']

    combined_df = combined_df[['player_name', 'player_commentary_z']].rename(columns={'player_commentary_z': 'player_commentary'})

    cap = pd.read_csv('pivot_final_2.csv')

    cap = cap[['name', 'year', 'percent_of_cap']].sort_values(by=['name', 'year']).drop_duplicates('name', keep='last')

    cap = cap.merge(combined_df, left_on='name', right_on='player_name')

    conditions = [
    (cap['percent_of_cap'] < 2.7),
    (cap['percent_of_cap'] >= 2.7) & (cap['percent_of_cap'] < 12.9),
    (cap['percent_of_cap'] >= 12.9)
    ]

    # create a list of the values we want to assign for each condition
    values = ['low', 'mid', 'high']

    # create a new column and use np.select to assign values to it using our lists as arguments
    cap['tier'] = np.select(conditions, values) 

    cap = cap.drop(columns='name')[['player_name', 'year', 'player_commentary', 'percent_of_cap', 'tier']]

    return cap

def clean_strings(string, exclude_words=[], extra_words=[]):    
    #Initialize NLP spacy object for lemmatization
    nlp = spacy.load('en_core_web_sm')
    
    #Basic clean
    lower_string = string.lower()
    
    normal_string = unicodedata.normalize('NFKD', lower_string)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    
    normal_no_chars_string = re.sub(r'[^a-z0-9\s]', '', normal_string.replace("'", " "))
    
    #Tokenize
    ttt = ToktokTokenizer()
    tokens = ttt.tokenize(normal_no_chars_string, return_str=True)
    
    doc = nlp(tokens)
    
    #Lemmatize
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    
    lemmatized_text
    
    #Remove stopwords
    stopword_list = stopwords.words('english')
    
    #Removing words from list
    stopword_list = [word for word in stopword_list if word not in exclude_words]
    
    #Adding words to list
    
    for word in extra_words:
        stopword_list.append(word)
    
    no_stop_words = [word for word in lemmatized_text.split() if word not in stopword_list]
    
    no_stop_string = ' '.join(no_stop_words)
    
    return no_stop_string

unigram_stopwords = [' game',
                    ' nfl',
                    ' season',
                    ' super',
                    ' bowl',
                    ' quarterback',
                    ' play',
                    ' team',
                    ' I',
                    ' performance',
                    ' year',
                    ' week',
                    'first',
                    'take', 
                    'last', 
                     'good',
                     'time',
                     'start',
                     'two',
                     'qb',
                     'new',
                     'throw',
                     'make', 
                     'pass',
                     'get',
                     'go', 
                     'touchdown',
                     'yard', 
                     'one', 
                     'win']

bi_tri_stopwords = [' I', ' blake',
 ' bortle',
 ' cody',
 ' kessler',
 ' mike',
 ' white',
 ' gardner',
 ' minshew',
 ' joe',
 ' flacco',
 ' philip',
 ' river',
 ' ryan',
 ' fitzpatrick',
 ' josh',
 ' rosen',
 ' drew',
 ' lock',
 ' deshaun',
 ' watson',
 ' ben',
 ' roethlisberger',
 ' sam',
 ' darnold',
 ' carson',
 ' palmer',
 ' matt',
 ' barkley',
 ' trevor',
 ' lawrence',
 ' taylor',
 ' heinicke',
 ' jalen',
 ' hurt',
 ' geno',
 ' smith',
 ' marcus',
 ' mariota',
 ' ryan',
 ' tannehill',
 ' patrick',
 ' mahome',
 ' tua',
 ' tagovailoa',
 ' justin',
 ' field',
 ' nick',
 ' mullen',
 ' baker',
 ' mayfield',
 ' david',
 ' blough',
 ' alex',
 ' smith',
 ' brock',
 ' purdy',
 ' daniel',
 ' jones',
 ' jared',
 ' goff',
 ' kirk',
 ' cousin',
 ' tyler',
 ' huntley',
 ' andrew',
 ' luck',
 ' jay',
 ' cutler',
 ' dwayne',
 ' haskin',
 ' aaron',
 ' rodger',
 ' russell',
 ' wilson',
 ' carson',
 ' wentz',
 ' matt',
 ' ryan',
 ' brock',
 ' osweiler',
 ' josh',
 ' mccown',
 ' colin',
 ' kaepernick',
 ' tyrod',
 ' taylor',
 ' cooper',
 ' rush',
 ' mac',
 ' jones',
 ' joe',
 ' burrow',
 ' drew',
 ' stanton',
 ' eli',
 ' manning',
 ' kyle',
 ' allen',
 ' josh',
 ' allen',
 ' dak',
 ' prescott',
 ' matthew',
 ' stafford',
 ' mitchell',
 ' trubisky',
 ' andy',
 ' dalton',
 ' brett',
 ' hundley',
 ' nick',
 ' fole',
 ' tom',
 ' brady',
 ' c.j.',
 ' beathard',
 ' sam',
 ' bradford',
 ' robert',
 ' griffin',
 ' iii',
 ' blaine',
 ' gabbert',
 ' mason',
 ' rudolph',
 ' justin',
 ' herbert',
 ' jamei',
 ' winston',
 ' jimmy',
 ' garoppolo',
 ' brian',
 ' hoyer',
 ' kyler',
 ' murray',
 ' brandon',
 ' allen',
 ' case',
 ' keenum',
 ' jacoby',
 ' brissett',
 ' bryce',
 ' petty',
 ' trevor',
 ' siemian',
 ' jeff',
 ' driskel',
 ' drew',
 ' bree',
 ' tom',
 ' savage',
 ' mike',
 ' glennon',
 ' deshone',
 ' kizer',
 ' teddy',
 ' bridgewater',
 ' zach',
 ' wilson',
 ' derek',
 ' carr',
 ' kenny',
 ' pickett',
 ' cam',
 ' newton',
 ' devlin',
 ' hodge',
 ' davis',
 ' mill',
 ' lamar',
 ' jackson',
 ' poll 0 vote quick link',
 ' follow share show comment',
 ' feedback thank first one comment',
 ' view post instagram',
 ' instagram instagram post',
 ' quote please credit',
 ' w w w',
 ' new england patriot',
 ' new orleans saint', 
 ' new york jet',
 ' kansas city chief',
 ' green bay packer',
 ' san francisco 49er',
 ' tampa bay buccaneer',
 ' los angeles ram',
 ' las vegas raider',
 ' new orleans saints',
 ' new york giants',
 ' new york jets',
 ' los angeles charger',
 ' washington football team',
 ' los angeles rams',
 ' san francisco 49ers',
 ' kansas city chiefs',
 ' carolina panther',
 ' denver broncos',
 ' pittsburgh steeler',
 ' chicago bear',
 ' philadelphia eagle',
 ' buffalo bill',
 ' houston texans',
 ' indianapolis colt',
 ' cleveland brown',
 ' 2023 arnold classic',
 ' mr olympia',
 ' miami dolphin',
 ' jacksonville jaguar',
 ' let take look',
 ' san diego charger',
 ' today2835 80420 yards12 ypa6 tdabsolute masterclass team',
 ' need 10ajmccarron post one great spring football',
 ' dallas cowboy',
 ' seattle seahawks',
 ' baltimore raven',
 ' washington commander',
 ' tennessee titans',
 ' detroit lion',
 ' arizona cardinal',
 ' lebron james',
 ' golden state warrior',
 ' new york giant',
 ' prime videoalso available',
 ' nfl nfl november 18 2022',
 ' gmt18 nov 2022',
 ' yard one touchdown',
 ' yard two touchdown',
 ' yard three touchdown',
 ' yard four touchdown',
 ' mccarron statline time performance',
 ' l l l',
 ' nation jason kelce say still beat', 
 ' sb loss speak still',
 ' twittercomjameslarsenpfnjame larsenjameslarsenpfnaj xfl',
 ' 2022 nfl season',
 ' 2021 nfl season',
 ' 2020 nfl season',
 ' 2022 nfl season',
 ' 20202021 nfl season',
 ' 2022 nfl draft',
 ' 2021 nfl draft',
 ' minnesota viking',
 ' ohio state',
 ' week 1']

def get_grams(df):
    high_words = ' '.join(df[df.tier == 'high'].player_commentary)
    mid_words = ' '.join(df[df.tier == 'mid'].player_commentary)
    low_words = ' '.join(df[df.tier == 'low'].player_commentary)

    unigram_high_words = high_words
    unigram_mid_words = mid_words
    unigram_low_words = low_words

    bi_tri_high_words = high_words
    bi_tri_mid_words = mid_words
    bi_tri_low_words = low_words   

    for word in unigram_stopwords:
        unigram_high_words = unigram_high_words.replace(word, '')
        unigram_mid_words = unigram_mid_words.replace(word, '')
        unigram_low_words = unigram_low_words.replace(word, '')

    for word in bi_tri_stopwords:
        bi_tri_high_words = bi_tri_high_words.replace(word, '')
        bi_tri_mid_words = bi_tri_mid_words.replace(word, '')
        bi_tri_low_words = bi_tri_low_words.replace(word, '')

    return unigram_high_words, unigram_mid_words, unigram_low_words, bi_tri_high_words, bi_tri_mid_words, bi_tri_low_words

def viz_unigrams(unigram_high_words, unigram_mid_words, unigram_low_words):
    plt.figure(figsize=(12,5))
    plt.suptitle('Top 20 Most Common Unigrams')

    plt.subplot(131)
    pd.Series(unigram_high_words.split()).value_counts().head(20).plot.barh()
    plt.title('High Percentage Caps')

    plt.subplot(132)
    pd.Series(unigram_mid_words.split()).value_counts().head(20).plot.barh()
    plt.title('Mid Percentage Caps')

    plt.subplot(133)
    pd.Series(unigram_low_words.split()).value_counts().head(20).plot.barh()
    plt.title('Low Percentage Caps')

    plt.tight_layout()
    plt.show()


def viz_bigrams(bi_tri_high_words, bi_tri_mid_words, bi_tri_low_words):
    plt.figure(figsize=(12,5))
    plt.suptitle('Most Common Bigrams')

    plt.subplot(131)
    top_20_bigrams_high = (pd.Series(nltk.ngrams(bi_tri_high_words.split(), 2))
                        .value_counts()
                        .head(20))
    top_20_bigrams_high[1:20].plot.barh()
    plt.title('High Percentage Caps')

    plt.subplot(132)
    top_20_bigrams_mid = (pd.Series(nltk.ngrams(bi_tri_mid_words.split(), 2))
                        .value_counts()
                        .head(20))
    top_20_bigrams_mid[1:20].plot.barh()
    plt.title('Mid Percentage Caps')

    plt.subplot(133)
    top_20_bigrams_low = (pd.Series(nltk.ngrams(bi_tri_low_words.split(), 2))
                        .value_counts()
                        .head(20))
    top_20_bigrams_low[1:20].plot.barh()
    plt.title('Low Percentage Caps')

    plt.tight_layout()
    plt.show()  

def viz_trigrams(bi_tri_high_words, bi_tri_mid_words, bi_tri_low_words):
    plt.figure(figsize=(12,5))
    plt.suptitle('Most Common Trigrams')

    plt.subplot(131)
    top_20_trigrams_high = (pd.Series(nltk.ngrams(bi_tri_high_words.split(), 3))
                        .value_counts()
                        .head(20))
    top_20_trigrams_high.plot.barh()
    plt.title('High Percentage Caps')

    plt.subplot(132)
    top_20_trigrams_mid = (pd.Series(nltk.ngrams(bi_tri_mid_words.split(), 3))
                        .value_counts()
                        .head(20))
    top_20_trigrams_mid.plot.barh()
    plt.title('Mid Percentage Caps')

    plt.subplot(133)
    top_20_trigrams_low = (pd.Series(nltk.ngrams(bi_tri_low_words.split(), 3))
                        .value_counts()
                        .head(20))
    top_20_trigrams_low.plot.barh()
    plt.title('Low Percentage Caps')

    plt.tight_layout()
    plt.show()

def get_sia_scores(df):

    sia = nltk.sentiment.SentimentIntensityAnalyzer()

    df['sentiment'] = df['player_commentary'].apply(lambda doc: sia.polarity_scores(doc)['compound'])

    return df.groupby('tier').mean('sentiment')