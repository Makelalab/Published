import logging
import os
import time
import pickle

from datetime import datetime
from pathlib import Path


def setup_logger(output_directory, analysis_id_short):
    log_file = os.path.join(output_directory.selected_path, 
                 f'{datetime.today().strftime("%Y%m%d")}_{analysis_id_short}.log')
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)
    
    return logger

def log_and_print(message):
    logging.info(message)
    print(message)

def save_as_pickle(data_to_save, output_directory, analysis_id_short, step):
    step_str = str(step).rjust(2, '0')
    pickle_file = os.path.join(output_directory.selected_path, 
                 f'{datetime.today().strftime("%Y%m%d")}_{analysis_id_short}_{step_str}.pickle')
    
    with open(pickle_file, 'wb') as f:
        pickle.dump(data_to_save, f)

    logging.info(f'data_saved to {pickle_file}')
    print(f'\ndata_saved to {pickle_file}')


def log_args_and_time(func):
    def wrapper(*args, **kwargs):
        log_and_print(f"Time before: {time.ctime()}")
        log_and_print(f"Called {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        log_and_print(f"Time after: {time.ctime()}")
        return result
    return wrapper