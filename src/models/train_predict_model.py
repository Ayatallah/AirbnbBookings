#imports
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


def read_data(filename):
    # Read File and Store its Content in Dataframe
    processed_data_path = os.path.join(os.path.pardir, 'data', 'processed')
    file_ = os.path.join(processed_data_path, filename)
    df_ = pd.read_csv(file_)
    return df_

def predict():
    train_df = read_data('train_users.csv')
    test_df = read_data('test_users.csv')
    test_id_df = test_df.id
    test_users_df = test_df.drop(columns=['id'])
    train_users_df = train_df.drop(columns=['id'])
    y_df = read_data('processed_train_target_data.csv')
    y_mapping_df = read_data('target_data_mapping.csv')
    y_mapping = {}
    for i in range(len(y_mapping_df)):
        y_mapping[i] = y_mapping_df.loc[i, 'country_destination']
    # Split training data into train and test
    X = train_df
    y = y_df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # RandomForestClassifier
    # Hyper parameter chosen by intuition beside trial and error
    clfRF = RandomForestClassifier(n_estimators=105, max_depth=25, random_state=0)
    clfRF.fit(train_df, y_df.values.ravel())
    y_test = clfRF.predict(test_df)
    columns = ['id', 'country']
    subm2_df = pd.DataFrame(columns=columns)
    subm2_df.id = test_id_df
    subm2_df.country = 'NDF'
    for i in range(len(y_test)):
        print(i)
        subm2_df.loc[i, 'country'] = str(y_mapping_df[y_test[i]])
    subm2_file = os.path.join(processed_data_path, 'subm2.csv')
    subm2_df.to_csv(subm2_file, index=False)

if __name__ == '__main__':
    predict()