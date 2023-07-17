from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import StandardScaler
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


    # metrics_train_df.rmse = metrics_train_df.rmse.astype(int)
    # metrics_validate_df.rmse = metrics_validate_df.rmse.astype(int)
    # metrics_test_df.rmse = metrics_test_df.rmse.astype(int)
    # print()
    # metrics_train_df.r2 = (metrics_train_df.r2 * 100).astype(int)
    # metrics_validate_df.r2 = (metrics_validate_df.r2 * 100).astype(int)
    # metrics_test_df.r2 = (metrics_test_df.r2 * 100).astype(int)

    return metrics_train_df, metrics_validate_df, metrics_test_df, predict_linear_train


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
    to_return['target'] = target

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
    return columns_list, target, corr_test


def new_visual_univariate_findings(df):
    for col in df.select_dtypes(include=['object']).columns:                   

        num_cols = len(df.select_dtypes(exclude=['object']).columns)
        num_rows, num_cols_subplot = divmod(num_cols, 3)
        if num_cols_subplot > 0:
            num_rows += 1

        fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))

        for i, col in enumerate(df.select_dtypes(exclude=['object']).columns):
            row_idx, col_idx = divmod(i, 3)
            sns.histplot(df[col], ax=axes[row_idx, col_idx])
            axes[row_idx, col_idx].set_title(f'Histogram of {col}')

        plt.tight_layout()
        plt.show()


def new_visual_multivariate_findings(df, target):
    for col in df.select_dtypes(include=['object']).columns:                   

        num_cols = len(df.select_dtypes(exclude=['object']).columns)
        num_rows, num_cols_subplot = divmod(num_cols, 3)
        if num_cols_subplot > 0:
            num_rows += 1

        fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))

        for i, col in enumerate(df.select_dtypes(exclude=['object']).columns):
            row_idx, col_idx = divmod(i, 3)
            sns.regplot(data = df, x = col, y = target, scatter_kws={'alpha': 0.1}, line_kws={'color': 'red'},ax=axes[row_idx, col_idx])
            
            axes[row_idx, col_idx].set_title(f'Scatterplot of {col} and {target}')

        plt.tight_layout()
        plt.show()