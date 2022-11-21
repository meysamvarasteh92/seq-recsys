import pandas as pd
import time
import datetime


def convert_to_timestamp(input_str: str) -> int:
    converted = time.mktime(datetime.datetime.strptime(
        input_str, '%Y-%m-%d %H:%M:%S.%f').timetuple())
    return int(converted)


def convert_table_to_str_files(csv_path: str, str_file_path: str):
    """ This function will convert the csv file into strings of history for each user.
        So if a user has watched courses 263, 347 and 401 in the same order, we will generate
        the string '263 347 401' for said user. There are 3 files generated:
        1. all.txt: This contains the complete history for each user
        2. train.txt: This contains all of the history for each user, except the last one
        3. test.txt: This contains the last interaction of each user, used for testing purposes

    Parameters
    ----------
    csv_path : str
        This will be the path to your .csv file containing Id, uval, course_id and start_date.
    str_file_path : path
        This is the path where the 3 text files will be saved and then used by mkhf.MKTrainer

    Examples
    --------
    >>> from mkhf.data_loader import convert_table_to_str_files
    >>> convert_table_to_str_files('data.csv', './textdata')
    """
    progress_data = pd.read_csv(csv_path)[['uval', 'course_id', 'start_date']]
    # Clear data
    progress_data = progress_data[progress_data['start_date'] != 'start_date']
    progress_data['uval'] = progress_data['uval'].astype(str)
    progress_data['start_date'] = progress_data['start_date'].astype(str)
    progress_data['start_date'] = progress_data['start_date'].map(
        lambda x: convert_to_timestamp(x))
    progress_data = progress_data.sort_values(by=['uval', 'start_date'])
    all_data = []
    train_data = []
    test_data = []
    current_sent = []
    current_user = progress_data.iloc[0]['uval']
    for row in progress_data.iterrows():
        item = row[1]
        if item['uval'] != current_user:
            all_data.append(' '.join(current_sent))
            if len(current_sent) >= 2:
                train_data.append(' '.join(current_sent[:-1]))
                test_data.append(current_sent[-1])
            current_sent = [item['course_id']]
            current_user = item['uval']
        else:
            current_sent.append(item['course_id'])

    with open(f'{str_file_path}/all.txt', 'w+') as f:
        for i in all_data:
            f.write(f'{i}\n')
    with open(f'{str_file_path}/train.txt', 'w+') as f:
        for i in train_data:
            f.write(f'{i}\n')
    with open(f'{str_file_path}/test.txt', 'w+') as f:
        for i in test_data:
            f.write(f'{i}\n')
