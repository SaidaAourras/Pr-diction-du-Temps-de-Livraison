import pandas as pd
# import numpy as np
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.feature_selection import SelectKBest , f_regression 
from sklearn.metrics import r2_score , mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# clean Data ( fillna ...)

def clean_data():
    data = pd.read_csv('dataset.csv')
    data['Weather'] = data['Weather'].fillna(data['Weather'].mode()[0])
    data['Traffic_Level'] = data['Traffic_Level'].fillna(data['Traffic_Level'].mode()[0])
    data['Time_of_Day'] = data['Time_of_Day'].fillna(data['Time_of_Day'].mode()[0])
    data['Courier_Experience_yrs'] =  data['Courier_Experience_yrs'].fillna(data['Courier_Experience_yrs'].mean())
    return data


# prepare data (encoder)

def prepare_data():
    data = clean_data()
    data = data.drop(columns=['Order_ID','Courier_Experience_yrs','Vehicle_Type'],axis=1)
    cols_cat = ['Weather','Traffic_Level','Time_of_Day']
    encoder = OneHotEncoder(sparse_output=False) # retourne un array NumPy dense au lieu d’une matrice creuse.
    encoded_data = encoder.fit_transform(data[cols_cat])
    cat_variables = encoder.get_feature_names_out(cols_cat)
    encoded_df = pd.DataFrame(encoded_data, columns=cat_variables)
    data = data.drop(columns=cols_cat)
    data = pd.concat([encoded_df,data],axis=1)
    return data

# print(prepare_data())
# split data to X , y

def split_X_Y():
    data = prepare_data()
    X = data.drop(columns=['Delivery_Time_min'])
    y = data['Delivery_Time_min']
    return X , y


# split Data train/test

def split_Train_Test_data():
    X , y = split_X_Y()
    X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train , X_test , y_train , y_test


# normalize X_test , X_train

def normalize_X_test_X_train():
    X_train , X_test , _ , _ = split_Train_Test_data()
    needed_cols = ["Distance_km" , "Preparation_Time_min"]
    scaler = StandardScaler()
    X_train[needed_cols] = scaler.fit_transform(X_train[needed_cols])
    X_test[needed_cols] = scaler.transform(X_test[needed_cols])
    
    return X_train , X_test

# use selectKBest

def select_best_cols():
    X_train , X_test = normalize_X_test_X_train()
    _ , _ , y_train , y_test =  split_Train_Test_data()
    selector = SelectKBest(score_func=f_regression,k=5)
    X_train_s = selector.fit_transform(X_train,y_train)
    X_test_s = selector.transform(X_test)
    
    return X_train_s , X_test_s , y_train , y_test

# Use (RandomForestRegressor or SVR)
# train model

def train_model(model):
    X_train_s , X_test_s , y_train , y_test = select_best_cols()
    # Grid Param
    
    grid_param = {
        RandomForestRegressor: {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
                },
        SVR:{
                'kernel': ['rbf', 'poly', 'linear'],
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'epsilon': [0.01, 0.1, 0.2, 0.5]
                }
    }
    
    model_type = type(model)
    
    # Grid Search CV
    
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid_param[model_type],
        scoring='r2',
        cv=5,
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_s,y_train)
    best_model = grid_search.best_estimator_
    y_predict = best_model.predict(X_test_s)

    r2 = r2_score(y_test , y_predict)
    mae = mean_absolute_error(y_test , y_predict)
    print(f'\nModel: {model_type.__name__}')
    print(f'Best Params: {grid_search.best_params_}')
    print(f'r2 : {r2} \nmean_absolute_error : {mae}')
    
    return best_model, r2, mae


# Example usage
# models = [RandomForestRegressor(random_state=42), SVR()]

# for model in models:
#     print(f"Training {type(model).__name__}...")
#     best_model, r2, mae = train_model(model)

# Bonus : use pipeline
def transform_num_cat_variables():
    data = clean_data()
    X = data.drop(columns=['Delivery_Time_min'])
    y = data['Delivery_Time_min']
    X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    cols_num = ["Distance_km" , "Preparation_Time_min"]
    cols_cat = ['Weather','Traffic_Level','Time_of_Day']
    
    transformer = ColumnTransformer([
    ('num', StandardScaler(),cols_num),
    ('cat', OneHotEncoder() ,cols_cat)
    ])
    
    return transformer , X_train , X_test , y_train , y_test
    
def training_model_with_Pipeline(model):
    transformer , X_train , X_test , y_train , y_test =  transform_num_cat_variables()
    pipeline = Pipeline([
    ('transformer',transformer),
    ('select',SelectKBest(score_func=f_regression)),
    ('model', model)
    ])

    # grid_param = {
    #     'model__n_estimators': [50, 100, 200],
    #     'model__max_depth': [5, 10, 15, None],
    #     'model__min_samples_split': [2, 5, 10],
    #     'model__min_samples_leaf': [1, 2, 4]
    # }
    
    grid_param = {
        'select__k':[5 , 8, 10 ,'all'],
        'model__kernel': ['rbf', 'linear'],
        'model__C': [1, 10],
        'model__gamma': ['scale', 0.1],
        'model__epsilon': [0.1, 0.2]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid_param,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=0
    )

    grid_search.fit(X_train , y_train)
    best_model = grid_search.best_estimator_
    y_predict = best_model.predict(X_test)
    # comparison = pd.DataFrame({
    # 'Valeur réelle': y_test,
    # 'Valeur prédite': y_predict
    # })

    # print(comparison)

    r2 = r2_score(y_test , y_predict)
    mae = mean_absolute_error(y_test , y_predict)
    print(f'\nModel: {model}')
    print(f'r2 : {r2} \nmean_absolute_error : {mae}')
    return r2 , mae
    
    

# print(f"Training {type(model).__name__}...")
# training_model_with_Pipeline(RandomForestRegressor(random_state=42))
# training_model_with_Pipeline(SVR())



# columns type  for test_pipeline
def type_columns(data):
    cat_col = data.select_dtypes(include='object').columns
    num_col = data.select_dtypes(include=['float64','int64']).columns
    return cat_col , num_col

    

