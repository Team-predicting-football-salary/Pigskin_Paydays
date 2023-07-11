import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

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
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def split_data(df, target):
    '''
    Takes in a dataframe and returns train, validate, test subset dataframes
    '''
    train, test = train_test_split(df,
                                   test_size=.2, 
                                   random_state=123, 
                                   stratify = df[target]
                                   )
    train, validate = train_test_split(train, 
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify = train[target]
                                       )
    
    return train, validate, test


def prepare_data(df):
    '''
    Prepares the data to be used in later functions
    '''
    df = df.drop(columns = ['CholCheck', 'PhysActivity', 'AnyHealthcare', 'NoDocbcCost','DiffWalk','GenHlth','Fruits','Veggies','Income','Education','Stroke'])

    df.columns = df.columns.str.lower()

    df['age'][df['age'] == 1] = '18 to 24'
    df['age'][df['age'] == 2] = '25 to 29'
    df['age'][df['age'] == 3] = '30 to 34'
    df['age'][df['age'] == 4] = '35 to 39'
    df['age'][df['age'] == 5] = '40 to 44'
    df['age'][df['age'] == 6] = '45 to 49'
    df['age'][df['age'] == 7] = '50 to 54'
    df['age'][df['age'] == 8] = '55 to 59'
    df['age'][df['age'] == 9] = '60 to 64'
    df['age'][df['age'] == 10] = '65 to 69'
    df['age'][df['age'] == 11] = '70 to 74'
    df['age'][df['age'] == 12] = '75 to 79'
    df['age'][df['age'] == 13] = '80 or older'


    df['sex'][df['sex'] == 1] = 'male'
    df['sex'][df['sex'] == 0] = 'female'
    df['bmi'] = df['bmi'][df['bmi'] < df['bmi'].quantile(.99)].copy()
    df = df.dropna()
    return df


def chi2_test(train, target, columns_list):
    '''
    Runs a chi2 test on all items in a list of lists and returns a pandas dataframe
    '''
    chi_df = pd.DataFrame({'feature': [],
                    'chi2': [],
                    'p': [],
                    'degf':[],
                    'expected':[]})
    
    for iteration, col in enumerate(columns_list):
        
        observed = pd.crosstab(train[target], train[col])
        chi2, p, degf, expected = stats.chi2_contingency(observed)

        chi_df.loc[iteration+1] = [col, chi2, p, degf, expected]

    return chi_df


def create_knn(X_train,y_train, X_validate, y_validate):
    '''
    creating a logistic_regression model
    fitting the logistic_regression model
    predicting the training and validate data
    '''
    the_df = pd.DataFrame(data=[
    {
        'model_train':'knn',
        'train_predict':229787/(229787+23893),
        'validate_predict':229787/(229787+23893),
        'n_neighbors': 'neighbors'
    }
    ])
    for i in range(20):
        knn = KNeighborsClassifier(n_neighbors=i+1)
        knn.fit(X_train, y_train)
        train_predict = knn.score(X_train, y_train)
        validate_predict = knn.score(X_validate, y_validate)
        the_df.loc[i+1] = ['KNeighborsClassifier', train_predict, validate_predict, i+1]

    return the_df


def create_logistic_regression(X_train,y_train, X_validate, y_validate):
    '''
    creating a logistic_regression model
    fitting the logistic_regression model
    predicting the training and validate data
    '''
    the_df = pd.DataFrame(data=[
    {
        'model_train':'LogisticRegression',
        'train_predict':229787/(229787+23893),
        'validate_predict':229787/(229787+23893),
        'C': 'the_c'
    }
    ])

    for iteration, i in enumerate([.01, .1, 1, 10, 100, 1000]):
        logit = LogisticRegression(random_state= 123,C=i)
        logit.fit(X_train, y_train)
        train_predict = logit.score(X_train, y_train)
        validate_predict = logit.score(X_validate, y_validate)
        the_df.loc[iteration + 1] = ['LogisticRegression', train_predict, validate_predict, i]

    return the_df


def create_random_forest(X_train,y_train, X_validate, y_validate,X_test, y_test):
    '''
    creating a random_forest model
    fitting the random_forest model
    predicting the training and validate data
    '''
    the_df = pd.DataFrame(data=[
    {
        'model_train':'RandomForestClassifier',
        'train_predict':229787/(229787+23893),
        'validate_predict':229787/(229787+23893),
        'max_depth': 'max_depth'
    }
    ])
    test_df = pd.DataFrame(data=[
    {
        'model_train':'RandomForestClassifier',
        'baseline':229787/(229787+23893),
        'max_depth': 'max_depth'
    }
    ])

    for i in range(20):
        forest = RandomForestClassifier(random_state = 123,max_depth=i +1 )
        forest.fit(X_train, y_train)    
        train_predict = forest.score(X_train, y_train)
        validate_predict = forest.score(X_validate, y_validate)
        the_df.loc[i + 1] = ['RandomForestClassifier', train_predict, validate_predict, i + 1]

    forest = RandomForestClassifier(random_state = 123,max_depth=9 )
    forest.fit(X_train, y_train)  
    test_predict = forest.score(X_test, y_test)
    test_df.loc[1] = ['RandomForestClassifier', round(test_predict, 3), 9]
    
    return the_df, test_df


def create_descision_tree(X_train,y_train, X_validate, y_validate):
    '''
    creating a Decision tree model
    fitting the Descision tree model
    predicting the training and validate data
    '''

    the_df = pd.DataFrame(data=[
    {
        'model_train':'DecisionTreeClassifier',
        'train_predict':2255/(2255+1267),
        'validate_predict':2255/(2255+1267),
        'max_depth': 'max_depth'
    }
    ])

    for i in range(20):

        tree = DecisionTreeClassifier(random_state = 123,max_depth= i + 1)
        tree.fit(X_train, y_train)
        train_predict = tree.score(X_train, y_train)
        validate_predict = tree.score(X_validate, y_validate)
        the_df.loc[i + 1] = ['DecisionTreeClassifier', train_predict, validate_predict, i + 1]

    return the_df


def scale_data(train,
               validate,
               test,
               cols = ['alcohol', 'density']):
    '''Takes in train, validate, and test set, and outputs scaled versions of the columns that were sent in as dataframes'''
    #Make copies for scaling
    train_scaled = train.copy() #Ah, making a copy of the df and then overwriting the data in .transform() to remove warning message
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #Initiate scaler, using MinMaxScaler
    scaler = MinMaxScaler()
    #Fit to train only
    scaler.fit(train[cols])
    #Creates scaled dataframes of train, validate, and test. This will still preserve columns that were not sent in initially.
    train_scaled[cols] = scaler.transform(train[cols])
    validate_scaled[cols] = scaler.transform(validate[cols])
    test_scaled[cols] = scaler.transform(test[cols])

    return train_scaled, validate_scaled, test_scaled


def mvp_info(train_scaled, validate_scaled, test_scaled,list_of_features, target):
    '''
    Takes in scaled data and a list of features to create the different feature and target variable objects
    '''
    X_train = train_scaled[list_of_features]
    X_validate = validate_scaled[list_of_features]
    X_test = test_scaled[list_of_features]


    y_train = train_scaled[target]
    y_validate = validate_scaled[target]
    y_test = test_scaled[target]

    return X_train, X_validate, X_test, y_train, y_validate, y_test


def get_dummies(X_train, X_validate, X_test,the_columns):
    '''
    Creates dummy columns for my catagorical variables 
    '''
    dummy_train = pd.get_dummies(X_train[the_columns])
    dummy_validate = pd.get_dummies(X_validate[the_columns])
    dummy_test = pd.get_dummies(X_test[the_columns])

    X_train = pd.concat([X_train, dummy_train], axis=1)
    X_validate = pd.concat([X_validate, dummy_validate], axis=1)
    X_test = pd.concat([X_test, dummy_test], axis=1)

    X_train = X_train.drop(columns =['sex', 'age'])
    X_validate = X_validate.drop(columns =['sex', 'age'])
    X_test = X_test.drop(columns =['sex', 'age'])

    return X_train, X_validate, X_test


def get_second_list(df):
    '''
    creates lists and a variable to be used in different functions
    '''
    the_list = list(df.columns)
    second_list = []
    target = the_list.pop(0)
    second_list.append(the_list.pop(-1))
    second_list.append(the_list.pop(2))
    second_list.append(the_list.pop(-3))
    second_list.append(the_list.pop(-2))
    the_age = second_list.pop(0)

    return second_list , the_age ,the_list, target


def calculate_percentage(value1, value2):
    '''
    Calculates the percentage of heart problem patients in the their respective catagories
    '''
    the_df = pd.DataFrame(data=[
    {
        'No heart problems':value1,
        'Heart problems':value2,
        'Percent heart problems':(value2 / (value1 + value2) ) * 100,
    }
    ])
    return the_df


def combine_three_dataframes(df1, df2, df3):
    '''
    Combines three dataframes
    '''
    return pd.concat([df1, df2, df3])


def combine_two_dataframes(df1, df2):
    '''
    combines two dataframes
    '''
    return pd.concat([df1, df2])


def the_order_list():
    '''
    creates an order to age
    '''
    the_order = ['18 to 24', '25 to 29', '30 to 34','35 to 39', '40 to 44',
            '45 to 49', '50 to 54', '55 to 59', '60 to 64', '65 to 69', 
            '70 to 74', '75 to 79', '80 or older']
    return the_order


def comparison_of_means(df, second_list):
    '''
    run a t test on items in a list and returns the results in a pandas dataframe
    '''
    df1 = pd.DataFrame()
    for i in second_list:
        t, p = stats.ttest_ind(df[i][df.heartdiseaseorattack == 1.0],df[i][df.heartdiseaseorattack == 0.0])
        df1 = pd.concat([df1, pd.DataFrame(data =[
            {
                'Category name':i,
                'P value':p
            }
        ])], axis = 0)
    return df1


def extra_analysis(df):
    '''
    creates a pandas dataframe for the extra analysis features and gives a percentage of those who have had 
    heart problems towards the whole
    '''
    label_mapping_1 = {'No High Blood Pressure':0 , 'High Blood Pressure':1}
    label_mapping_2 = {'Normal Cholesterol':0 ,  'High Cholesterol':1}
    label_mapping_3 = {'Non-Smoking':0 ,  'Smoking':1}
    label_mapping_4 = { 'No Diabetes':0,  'Pre-Diabetes':1, 'Diabetes':2}
    label_mapping_5 = { 'No hvyalcoholconsump':0,  'hvyalcoholconsump':1}
    
    df['highbp'] = df['highbp'].map(label_mapping_1)
    df['highchol'] = df['highchol'].map(label_mapping_2)  
    df['smoker'] = df['smoker'].map(label_mapping_3)
    df['diabetes'] = df['diabetes'].map(label_mapping_4)
    df['hvyalcoholconsump'] = df['hvyalcoholconsump'].map(label_mapping_5)


    df1 = calculate_percentage(df.heartdiseaseorattack[df.diabetes == 0].value_counts()[0], df.heartdiseaseorattack[df.diabetes == 0].value_counts()[1])
    df2 = calculate_percentage(df.heartdiseaseorattack[df.diabetes == 1].value_counts()[0], df.heartdiseaseorattack[df.diabetes == 1].value_counts()[1])
    df3 = calculate_percentage(df.heartdiseaseorattack[df.diabetes == 2].value_counts()[0], df.heartdiseaseorattack[df.diabetes == 2].value_counts()[1])
    diabetes_df = combine_three_dataframes(df1, df2, df3)
    diabetes_df.reset_index(drop=True, inplace=True)
    diabetes_df = diabetes_df.rename(index={0: 'No diabetes'})  
    diabetes_df = diabetes_df.rename(index={1: 'Pre diabetes'})
    diabetes_df = diabetes_df.rename(index={2: 'Has diabetes'})

    df1 = calculate_percentage(df.heartdiseaseorattack[df.hvyalcoholconsump == 0].value_counts()[0], df.heartdiseaseorattack[df.hvyalcoholconsump == 0].value_counts()[1])
    df2 = calculate_percentage(df.heartdiseaseorattack[df.hvyalcoholconsump == 1].value_counts()[0], df.heartdiseaseorattack[df.hvyalcoholconsump == 1].value_counts()[1])
    alcohol_df = combine_two_dataframes(df1, df2)
    alcohol_df.reset_index(drop=True, inplace=True)
    alcohol_df = alcohol_df.rename(index={0: 'Alcohol free'})
    alcohol_df = alcohol_df.rename(index={1: 'Heavy alcohol'})

    df1 = calculate_percentage(df.heartdiseaseorattack[df.sex == 'male'].value_counts()[0], df.heartdiseaseorattack[df.sex == 'male'].value_counts()[1])
    df2 = calculate_percentage(df.heartdiseaseorattack[df.sex == 'female'].value_counts()[0], df.heartdiseaseorattack[df.sex == 'female'].value_counts()[1])
    gender_df = combine_two_dataframes(df1, df2)
    gender_df.reset_index(drop=True, inplace=True)
    gender_df = gender_df.rename(index={1: 'Male'})
    gender_df = gender_df.rename(index={0: 'Female'})

    the_df = combine_three_dataframes(diabetes_df, alcohol_df, gender_df)
    return the_df


def load_all_csv():
    '''
    loads all model csv's and concatenates the best results together
    '''
    knn_df = pd.read_csv('KNN.csv', index_col=0)
    LR_df = pd.read_csv('LR.csv',index_col=0)
    RF_df = pd.read_csv('RF.csv',index_col=0)
    DT_df = pd.read_csv('DT.csv',index_col=0)
    testRF = pd.read_csv('testRF.csv',index_col=0)

    knn_df.iloc[1:2]
    LR_df.iloc[1:2]
    RF_df.iloc[9:10]
    DT_df.iloc[1:2]
    the_df = pd.concat([knn_df.iloc[1:2], LR_df.iloc[1:2], RF_df.iloc[9:10], DT_df.iloc[1:2]])
    return the_df, testRF


def multivariate_exploration_charts(df, the_list, second_list):
    '''
    creates barplots for all possible combinations of continuous varables with catagorical variables
    '''
    label_mapping_1 = {0:'No High Blood Pressure' , 1:'High Blood Pressure'}
    label_mapping_2 = {0:'Normal Cholesterol' ,  1:'High Cholesterol'}
    label_mapping_3 = {0:'Non-Smoking' ,  1:'Smoking'}
    label_mapping_4 = {0:'No Diabetes',  1:'Pre-Diabetes', 2:'Diabetes'}
    label_mapping_5 = {0:'No hvyalcoholconsump',  1:'hvyalcoholconsump'}
    
    df['highbp'] = df['highbp'].map(label_mapping_1)
    df['highchol'] = df['highchol'].map(label_mapping_2)
    df['smoker'] = df['smoker'].map(label_mapping_3)
    df['diabetes'] = df['diabetes'].map(label_mapping_4)
    df['hvyalcoholconsump'] = df['hvyalcoholconsump'].map(label_mapping_5)

    plt.figure(figsize=(14,14))
    plt.xticks(rotation = 45)
    i = 0

    the_list # catagorical
    second_list # continuous
    charts_list = [['bmi', 'highbp'], ['menthlth','highbp'], ['physhlth', 'highbp'],
     ['menthlth', 'highchol'], ['physhlth', 'highchol'],
     ['menthlth','smoker'], ['physhlth', 'smoker'], 
     ['bmi', 'hvyalcoholconsump'], ['menthlth','hvyalcoholconsump'], ['physhlth','hvyalcoholconsump'],
     ['menthlth', 'sex'], ['physhlth', 'sex']]
    
    # for col in the_list:
    #     for second in second_list: 
    #         plt.subplot(7,3,i+1)
    #         sns.barplot(data=df, x=col, y=second, hue=df.heartdiseaseorattack).set_title(f'{col}')
    #         i +=1
    for col in charts_list:
        plt.subplot(7,3,i+1)
        sns.barplot(data=df, x=df[col[1]], y=df[col[0]], hue=df.heartdiseaseorattack).set_title(f'{col[0]} and {col[1]}')
        i +=1
    plt.tight_layout()
    plt.show()


def bivariate_catagorical(df, the_list):
    '''
    creates countplots and adds them to a subplot for all of my catagorical variables
    '''
    label_mapping_1 = {0: 'No High Blood Pressure', 1: 'High Blood Pressure'}
    label_mapping_2 = {0: 'Normal Cholesterol', 1: 'High Cholesterol'}
    label_mapping_3 = {0: 'Non-Smoking', 1: 'Smoking'}
    label_mapping_4 = {0: 'No Diabetes', 1: 'Pre-Diabetes', 2:'Diabetes'}
    label_mapping_5 = {0: 'No hvyalcoholconsump', 1: 'hvyalcoholconsump'}
    
    df['highchol'] = df['highchol'].map(label_mapping_2)
    df['highbp'] = df['highbp'].map(label_mapping_1)
    df['smoker'] = df['smoker'].map(label_mapping_3)
    df['diabetes'] = df['diabetes'].map(label_mapping_4)
    df['hvyalcoholconsump'] = df['hvyalcoholconsump'].map(label_mapping_5)
    
    plt.figure(figsize=(14,14))
    plt.xticks(rotation = 45)
    for i, col in enumerate(the_list):
        plt.subplot(4,3,i+1)
        sns.countplot(hue=df['heartdiseaseorattack'], x=df[col], data=df).set_title(f'{col}')
        if col in ['highbp', 'highchol', 'hvyalcoholconsump']:
            plt.xticks(rotation = 10)

    plt.subplots_adjust(wspace=0.5, hspace=0.75)
    plt.show()


def age_visual(df, the_age):
    '''
    creates a barplot for the age catagory
    '''
    the_order = the_order_list()
    sns.countplot(hue=df['heartdiseaseorattack'], x=df['age'], data=df, order = the_order).set_title(f'{the_age}')
    plt.xticks(rotation=45)
    plt.show()


def bivariate_continuous(df, second_list):
    '''
    creates violinplots of all of my continuous variables
    '''
   
    plt.figure(figsize=(14,14))
    plt.xticks(rotation = 45)
    for i, col in enumerate(second_list):
        plt.subplot(4,3,i+1)
        sns.violinplot(x=df['heartdiseaseorattack'], y=df[col], data=df).set_title(f'{col}')
        
        mean_value = np.mean(df[col][df['heartdiseaseorattack'] == 1])
        mean_value_2 = np.mean(df[col][df['heartdiseaseorattack'] == 0])
        plt.axhline(mean_value, color='red', linestyle='--', label='Mean heart attack')
        plt.axhline(mean_value_2, color='blue', linestyle='--', label='Mean NoN heart attack')
        plt.xticks(rotation = 10, ticks = [0, 1], labels=['No heart problems', 'Heart problems'])
        plt.legend()
        
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()
   

def univariate_visual(df):
    '''
    creates histplots for all of my columns
    '''
    plt.figure(figsize=(14,14))
    plt.xticks(rotation = 45)
    for i, col in enumerate(df):
        plt.subplot(4,3,i+1)
    
        sns.histplot(df[col])
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    plt.show()


def condenced_prepate(df):
    '''
    runs multiple functions that prepare the data
    '''
    df = prepare_data(df)
    second_list, the_age, the_list, target = get_second_list(df)
    train, validate, test = split_data(df, 'heartdiseaseorattack')
    train_scaled, validate_scaled, test_scaled = scale_data(train, validate, test, cols = second_list)
    return df, second_list, the_age, the_list, target, train, validate, test , train_scaled, validate_scaled, test_scaled

def create_random_forest_two(X_train,y_train, X_validate, y_validate,X_test, y_test):
    '''
    gets the weights of the best model
    '''
    
    forest = RandomForestClassifier(random_state = 123,max_depth=9 )
    forest.fit(X_train, y_train)    
    train_predict = forest.score(X_train, y_train)
    validate_predict = forest.score(X_validate, y_validate)
    the_weights = forest.feature_importances_
    
    return forest


def getting_weights_max(tree, X_train):
    columns = X_train.columns

    the_weight = tree.feature_importances_
    the_weight
    weights_column = []
    for i in the_weight:
        weights_column.append(i)
        
    the_dataframe = pd.DataFrame({'columns': columns, 
                                'the_weight':weights_column})  

    the_dataframe
    plt.title('Does device_protection affect churn')  

    ax = sns.barplot(x=columns , y=the_weight, data = the_dataframe)
    ax.tick_params(axis='x', rotation=90)
    plt.show()


def correlation_charts(train,columns_list, target):
    '''
    Creates and shows visuals for Correlation tests 
    '''
    plt.figure(figsize=(14,3))
    plt.suptitle('Bivariate Exploration: The Strongest Correlators of Wine Quality')
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
        corr_df.loc[i] = [col, abs(r), p]
    to_return = corr_df.sort_values(by='r', ascending=False)
    to_return['target'] = target
    return to_return
    