import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd


def update_database(conditions, xml_path):
# update master csv

# conditions_list = ['name',
#                    'date', 
#                    'dye', 
#                    'species', 
#                    'strain', 
#                    'temperature', 
#                    'medium', 
#                    'chemical', 
#                    'induction',
#                    'comment',
#                ]
    
# df = pd.DataFrame(columns = (conditions_list+['git_hash', 'uuid_experiment', 'uuid_file', 'uuid_analysis', 'path']))
# df.to_csv(os.getenv('SMPP_DATABASE'), sep = '\t', index = False)

    df = pd.read_csv(os.getenv('SMPP_DATABASE'), sep = '\t')
    new_data = pd.DataFrame(columns = df.columns)
    for i in new_data.columns:
        try:
            new_data[i] = [conditions[i]]
        except KeyError:
            pass
    new_data['path'] = xml_path
    df = pd.concat([df, new_data], ignore_index = True)
    df.to_csv(os.getenv('SMPP_DATABASE'), sep = '\t', index = False)

    print('that\'s it, database updated:')
    return(df)

def update_db_img(exp_meta_df, file_comment, file_git_hash, file_uuid, bucket_name, object_key):
    
    '''
    Index(['name', 'date', 'dye', 'species', 'strain', 'temperature', 'medium',
       'chemical', 'induction', 'comment', 'git_hash', 'uuid_experiment',
       'uuid_file', 'uuid_analysis', 'path'],
      dtype='object')
    '''

    df = pd.read_csv(os.getenv('SMPP_DATABASE'), sep = '\t')
    
    new_data = exp_meta_df.copy()
    new_data.loc[new_data.index[-1], 'comment'] = file_comment
    new_data.loc[new_data.index[-1], 'git_hash'] = file_git_hash
    new_data.loc[new_data.index[-1], 'uuid_file'] = file_uuid
    new_data.loc[new_data.index[-1], 'path'] = str((bucket_name, object_key)) # s3_resource.Object(*eval(df['path'])) # this shuld unpack the tuple to arguments 

    df = pd.concat([df, new_data], ignore_index = True)
    df.to_csv(os.getenv('SMPP_DATABASE'), sep = '\t', index = False)

    print('that\'s it, database updated:')
    return(df)

def update_db_analysis(exp_meta_df, analysis_comment, analysis_git_hash, analysis_uuid, analysis_path, type):
    
    '''
    Index(['name', 'date', 'dye', 'species', 'strain', 'temperature', 'medium',
       'chemical', 'induction', 'comment', 'git_hash', 'uuid_experiment',
       'uuid_file', 'uuid_analysis', 'path'],
      dtype='object')
    '''
    if type == 'mask':
        
        df = pd.read_csv(os.getenv('SMPP_DATABASE'), sep = '\t')
        
        new_data = exp_meta_df.copy()
        new_data.loc[new_data.index[-1], 'comment'] = analysis_comment
        new_data.loc[new_data.index[-1], 'git_hash'] = analysis_git_hash
        new_data.loc[new_data.index[-1], 'uuid_analysis'] = analysis_uuid
        new_data.loc[new_data.index[-1], 'path'] = analysis_path 
    
        df = pd.concat([df, new_data], ignore_index = True)
        df.to_csv(os.getenv('SMPP_DATABASE'), sep = '\t', index = False)
    
        print('that\'s it, database updated:')
        return(df)

    else:
        return 'unknown analysis type'


def csv_editor(action_w, data, selected_df):
   
    col = action_w.children[0].value
    idx = action_w.children[1].value
    new = action_w.children[2].value
    dl  = action_w.children[3].value
    
    
    # print(col)
    # print(idx)
    # print(new)
    # print(dl )
    
    
    dbcopy = data.copy()
    
    if idx.isnumeric():
        # dbcopy.loc[int(idx), col] = new_value
        print('Ok. Here is the data you chose:')
        display(dbcopy.loc[[int(idx)], [col]])
        idx = dbcopy.loc[[int(idx)], [col]].index
        
    elif ',' in idx:
        idx = selected_df.loc[[int(i) for i in idx.replace(',', ' ').split()]].index
        print('Ok. Here is the data you chose:')
        display(dbcopy.loc[idx, [col]])
     
    
    elif ('-' in idx) and (len([int(i) for i in idx.replace('-', ' ').split()]) == 2):  # range
        idx = selected_df.loc[
              [int(i) for i in idx.replace('-', ' ').split()][0]:          # range from
              [int(i) for i in idx.replace('-', ' ').split()][1]           # to
                          ].index
        print('Ok. Here is the data you chose:')
        display(dbcopy.loc[idx, [col]])  
    
    else:
        return 'Huh? unclear input, try different indices in the cell above'
    
    
    
    # correct or mark for deletion
    if not dl:
        if new:
            print('Corrected version:')
            dbcopy.loc[idx, col] = new
            display(dbcopy.loc[idx])
        else:
            return 'what is the desired value? type in the cell above'
    
    else:
    
        print(f"\033[92m!!! This data will be DELETED !!!\033[0m")
        print('note that if you delete an experiment all \"children\" files will be deleted\nsame is true for masks associated with the deleted tiff(s)')
        print('')
        
        inst = dbcopy.loc[idx]
    
        # uuids to delete:
    
        exp_to_del   = inst[inst['uuid_file'].isna()&inst['uuid_analysis'].isna()]['uuid_experiment'].to_list()
        tiffs_to_del = inst[inst['uuid_file'].notna()&inst['uuid_analysis'].isna()]['uuid_file'].to_list()
        masks_to_del = inst[inst['uuid_analysis'].notna()]['uuid_analysis'].to_list()
        
        # print(exp_to_del, tiffs_to_del, masks_to_del)
        
        for e in exp_to_del:
            idx = dbcopy.loc[dbcopy['uuid_experiment']==e].index
            dbcopy.loc[idx, 'comment'] = '[DELETE] ' + dbcopy.loc[idx, 'comment'].astype(str)
            display(dbcopy.loc[idx])
            print('')
        
        for t in tiffs_to_del:
            idx = dbcopy.loc[dbcopy['uuid_file']==t].index
            dbcopy.loc[idx, 'comment'] = '[DELETE] ' + dbcopy.loc[idx, 'comment'].astype(str)
            display(dbcopy.loc[idx])
            print('')
        
        for m in masks_to_del:
            idx = dbcopy.loc[dbcopy['uuid_analysis']==m].index
            dbcopy.loc[idx, 'comment'] = '[DELETE] ' + dbcopy.loc[idx, 'comment'].astype(str)
            display(dbcopy.loc[idx])
            print('')
    
    
    decision = input('is that what you want? type \'yes\' to confirm')
    
    if decision == 'yes':
        ts = str(int(datetime.timestamp(datetime.now()))) # datetime.fromtimestamp(int(ts), tz=None)
        shutil.copyfile(os.getenv('SMPP_DATABASE'), f'{os.getenv("SMPP_DATA_DIR")}/temp_backups/database_backup_{ts}.tsv')
        
        dbcopy.to_csv(os.getenv('SMPP_DATABASE'), sep = '\t', index = False)
        print('done, database is updated')
    
    else:
        print('rerun cell above')


def h5_update_db_img(exp_meta_df, file_comment, file_git_hash, file_uuid, fov_uuid, bucket_name, object_key):
    
    '''
    Index(['name', 'date', 'dye', 'species', 'strain', 'temperature', 'medium',
       'chemical', 'induction', 'comment', 'git_hash', 'uuid_experiment', 'uuid_fov',
       'uuid_file', 'uuid_analysis', 'path'],
      dtype='object')
    '''

    df = pd.read_csv(os.getenv('SMPP_DATABASE'), sep = '\t')
    
    new_data = exp_meta_df.copy()
    new_data.loc[new_data.index[-1], 'comment'] = file_comment
    new_data.loc[new_data.index[-1], 'git_hash'] = file_git_hash
    new_data.loc[new_data.index[-1], 'uuid_file'] = file_uuid
    new_data.loc[new_data.index[-1], 'uuid_fov'] = fov_uuid
    new_data.loc[new_data.index[-1], 'path'] = str((bucket_name, object_key)) # s3_resource.Object(*eval(df['path'])) # this shuld unpack the tuple to arguments 

    
    df = pd.concat([df, new_data], ignore_index = True)
    df.to_csv(os.getenv('SMPP_DATABASE'), sep = '\t', index = False)
    
    return(df)

def show_df_style(df):
    unique_uuids = []
    for i in ['uuid_experiment', 'uuid_fov', 'uuid_file', 'uuid_analysis']:
        unique_uuids = unique_uuids + df[i].unique().tolist()
    
    
    colors = cm.get_cmap('RdYlBu', len(unique_uuids))  # Choose a colormap
    color_mapping = {uuid: colors(i) for i, uuid in enumerate(unique_uuids)}
    
    def highlight_uuids(s):
        return [f'background-color: rgba({int(color_mapping[uuid][0]*255)}, {int(color_mapping[uuid][1]*255)}, {int(color_mapping[uuid][2]*255)}, 1)' for uuid in s]
    
    display(df[['name', 'date', 'comment', 'uuid_experiment',
           'uuid_fov', 'uuid_file', 'uuid_analysis']].style.apply(highlight_uuids, subset=['uuid_experiment',
           'uuid_fov', 'uuid_file', 'uuid_analysis']).set_properties(**{'color': 'black'}, subset=['uuid_experiment',
           'uuid_fov', 'uuid_file', 'uuid_analysis']))


