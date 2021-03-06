#imports
import pandas as pd
import numpy as np
import os
import inspect
from datetime import datetime, timedelta

base_file_path = inspect.getframeinfo(inspect.currentframe()).filename
base_path = os.path.dirname(os.path.abspath(base_file_path))
project_dir_path = os.path.join(base_path, '../..')

processed_data_path = os.path.join(project_dir_path, 'data', 'processed')
raw_data_path = os.path.join(project_dir_path, 'data', 'raw')

def read_data(filename):
    # Read File and Store its Content in Dataframe
    file_ = os.path.join(raw_data_path, filename)
    df_ = pd.read_csv(file_)
    return df_

def write_data(df_, filename):
    # Write DataFrame Content in File
    file_ = os.path.join(processed_data_path, filename)
    df_.to_csv(file_, index=False)

def prepare_train_test_data(train_df, test_df):
    print("Preprocess Data: prepare_train_test_data Started")
    #Convert Date/Time Fields from object to datetime
    # In train, convert date_account_created,timestamp_first_active,date_first_booking
    train_df['date_account_created'] = pd.to_datetime(train_df['date_account_created'])
    train_df['timestamp_first_active'] = pd.to_datetime(train_df['timestamp_first_active'])
    train_df['date_first_booking'] = pd.to_datetime(train_df['date_first_booking'])
    # Convert Date/Time Fields from object to datetime
    # In test, convert only timestamp_first_active, date_account_created
    # In test, date_first_booking is always NA since its attached to the first booking to be predicted
    test_df['timestamp_first_active'] = pd.to_datetime(test_df['timestamp_first_active'])
    test_df['date_account_created'] = pd.to_datetime(test_df['date_account_created'])
    # Drop row where user signed up after first booking, only 29 rows
    train_df = train_df.drop(train_df[train_df['date_first_booking'] < train_df['date_account_created']].index)
    #print(len(train_df))
    # Extract Target value
    country_destination = train_df['country_destination']
    # Drop Target value
    train_df = train_df.drop(columns=['country_destination'])
    # Concatenating train and test dfs to manipulate same columns all at once
    data_df = pd.concat([train_df, test_df], ignore_index=True)
    #print(data_df.info())
    return data_df, country_destination, train_df


def impute_users_data(data_df):
    print("Preprocess Data: impute_users_data Started")
    # Drop 1st booking date
    data_df = data_df.drop(columns=['date_first_booking'])
    # Imputer Age column outliers
    data_df.loc[data_df['age'] > 115, ['age']] = data_df['age'].median()
    data_df.loc[data_df['age'] < 10, ['age']] = data_df['age'].median()
    # Fill in Age column missing values
    data_df['age'].fillna(data_df['age'].mean(), inplace=True)
    # Fill in 1st affiliate tracked missing values
    data_df['first_affiliate_tracked'].fillna(str(data_df['first_affiliate_tracked'].mode()[0]), inplace=True)
    # Introduce season_accnt_crtd field holding data about seasons when accounts were created
    data_df.loc[:, ('season_accnt_crtd')] = ""
    for i in range(275518):
        print("impute "+str(i))
        month = int(data_df['date_account_created'][i].strftime("%m"))
        if (month == 1 or month == 2 or month == 12):
            data_df.loc[i, ('season_accnt_crtd')] = "winter"
        if (month == 3 or month == 4 or month == 5):
            data_df.loc[i, ('season_accnt_crtd')] = "spring"
        if (month == 6 or month == 7 or month == 8):
            data_df.loc[i, ('season_accnt_crtd')] = "summer"
        if (month == 9 or month == 11 or month == 10):
            data_df.loc[i, ('season_accnt_crtd')] = "automn"
    # Drop datetime fields
    data_df = data_df.drop(columns=['date_account_created'])
    data_df = data_df.drop(columns=['timestamp_first_active'])
    # Encode Gender based on whether its known or not (unknown-other)
    data_df['gender_known'] = -1
    data_df.loc[data_df.gender == "MALE", ['gender_known']] = 1
    data_df.loc[data_df.gender == "FEMALE", ['gender_known']] = 1
    data_df.loc[data_df.gender == "OTHER", ['gender_known']] = 0
    data_df.loc[data_df.gender == "-unknown-", ['gender_known']] = 0
    # Encode Gender based on male or not(male-female)
    data_df['is_male'] = 0
    data_df.loc[data_df.gender == "MALE", ['is_male']] = 1
    data_df = data_df.drop(columns=['gender'])
    #writee to data_users_file = os.path.join(raw_data_path, 'data_users.csv')
    return data_df

# This method is to be optimized
def process_sessions(sessions_df):
    print("Preprocess Data: process_sessions Started")
    # Processing seemingly unnecessary columns
    sessions_df = sessions_df.drop(columns=['action_type', 'action_detail', 'secs_elapsed', 'device_type'])
    # Process sessions_df so that each user has one row showing all his behaviour
    s_group = sessions_df.groupby(['user_id'])
    users_have_session = sessions_df.user_id.unique().tolist()
    users_have_session = [x for x in users_have_session if str(x) != 'nan']
    actions = sessions_df['action'].unique().tolist()
    columns = ['id']+actions
    df_ = pd.DataFrame( columns=columns)
    df_.id = users_have_session
    df_.fillna(0, inplace=True)
    for i in range(len(users_have_session)):
        print("sessions "+str(i))
        user_actions = s_group.get_group(str(users_have_session[i])).action.unique().tolist()
        user_actions = [x for x in user_actions if str(x) != 'nan']
        for k in range(len(user_actions)):
            action_count = s_group.get_group(users_have_session[i]).action.value_counts()[str(user_actions[k])]
            df_.loc[df_['id']==users_have_session[i], [user_actions[k]]] = action_count
    return df_

def prepare_user_data(data_df, users_actions_df):
    print("Preprocess Data: prepare_user_data Started")
    # Merge data_df & users_action_df so that each user has 1 row showing info & behaviour
    check_user_session = pd.merge(data_df, users_actions_df, on='id', how='left')
    # Fill missings columns with 0 since session data were not available for all users
    check_user_session.fillna(0, inplace=True)
    # Encode all categorical data using one-hot-encoding
    check_user_session_encoded = pd.get_dummies(check_user_session,
                                                columns=['signup_method', 'signup_flow', 'language', 'signup_app',
                                                         'first_device_type', 'first_browser', 'season_accnt_crtd',
                                                         'affiliate_channel', 'first_affiliate_tracked',
                                                         'affiliate_provider'])
    check_user_session_encoded['session_known'] = 0
    check_user_session_encoded.loc[check_user_session_encoded['id'].isin(users_actions_df.id), 'session_known'] = 1
    return check_user_session_encoded


def process_target(country_destination):
    print("Preprocess Data: process_target Started")
    # Encode target variable
    factor = pd.factorize(country_destination)
    processed_y = pd.DataFrame(columns=['country_destination'])
    processed_y.country_destination = factor[0]
    mappings = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    processed_y_mapping = pd.DataFrame( columns=['factor','country_destination'])
    processed_y_mapping.factor = mappings
    processed_y_mapping.country_destination = factor[1]
    write_data(processed_y,'processed_train_target_data.csv')
    write_data(processed_y_mapping, 'target_data_mapping.csv')

def preprocess_raw_data():
    print("Preprocess Data: preprocess_raw_data Started")
    train_df = read_data('train_users_2.csv')
    test_df = read_data('test_users.csv')
    data_df, country_destination, train_df = prepare_train_test_data(train_df, test_df)
    data_df = impute_users_data(data_df)
    write_data(data_df, 'data_users.csv')
    sessions_df = read_data('sessions.csv')
    #The next step took 6 hours running on 4Core Pc
    user_actions = process_sessions(sessions_df) #It needs optimization, till done, the output file is made available in \data
    write_data(user_actions,'users_actions.csv' )
    users_actions_df =  pd.read_csv(os.path.join(processed_data_path, 'users_actions.csv'))
    all_users_data = prepare_user_data(data_df, users_actions_df)
    write_data(all_users_data, 'all_users_processed_data.csv')
    ##here is the problem, debugged
    processed_train_df = all_users_data[:len(train_df)]
    processed_test_df = all_users_data[len(train_df):]
    write_data(processed_train_df, 'train_users.csv')
    write_data(processed_test_df, 'test_users.csv')
    process_target(country_destination)

if __name__ == '__main__':
    print("Preprocess Data: Main Started")
    preprocess_raw_data()
    print("Preprocess Data: Main Ended")