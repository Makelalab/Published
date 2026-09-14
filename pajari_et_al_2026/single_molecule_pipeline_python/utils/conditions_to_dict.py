import uuid
import subprocess
from datetime import datetime
import numpy as np
import h5py


def get_dict(conditions_list, widgets_list):

    def get_git_hash():
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

    conditions_dict = {}
    for condition, widget in zip(conditions_list, widgets_list):
        if widget.value == 'other / ours / value':
            conditions_dict[condition] = input('specify {}:'.format(condition))
            
        else:
            conditions_dict[condition] = widget.value
    
    # these are user input-independent:
    conditions_dict['git_hash'] = get_git_hash()
    conditions_dict['uuid_experiment'] = str(uuid.uuid1())

    return conditions_dict


def conditions_to_df(database, widgets_list, fields, entry_type):
    

    filtered = 'nothing to show'
    if entry_type=='Experiments':
        filtered = database[database['uuid_file'].isna()&database['uuid_analysis'].isna()]
    elif entry_type=='Raw images':
        filtered = database[database['uuid_file'].notna()&database['uuid_analysis'].isna()]
    elif entry_type=='Analysis':
        filtered = database[database['uuid_analysis'].notna()]
    else:
        filtered = database
    
    values = [str(i.value) for i in widgets_list]
    
    for j,k in zip(fields[:-2], values[:-2]):
        #print(j, k)
        if k in ['None', 'any']:
            pass
        else:
            filtered = filtered.where(database[j]==k).dropna(how='all')
    
    if widgets_list[-2].value=='':
        pass
    else:
        filtered = filtered.drop([int(i) for i in widgets_list[-2].value.replace(',', ' ').split()], errors = 'ignore')
    
    if widgets_list[-1].value=='':
        pass
    else:
        filtered = filtered.loc[[int(i) for i in widgets_list[-1].value.replace(',', ' ').split()]]

    if filtered.empty:
        filtered = 'nothing to show'


    return filtered


def get_fovs(database, experiment_id):
       
    raw_images = database[database['uuid_file'].notna()&database['uuid_analysis'].isna()]
    raw_images = raw_images.where(raw_images['uuid_experiment']==experiment_id).dropna(how='all', axis=0)
    
    masks = database[database['uuid_analysis'].notna()]
    masks = masks.loc[masks['path'].str.contains('.npy|.png', regex=True)]
    masks = masks.loc[masks['uuid_file'].isin(raw_images['uuid_file'].values)]

    return raw_images, masks


def file_meta(df_selected):

    if len(df_selected.index) != 1:
        return 'you have to select one experiment, repeat the selection'
        
    else:
        def get_git_hash():
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    
        free_form_comment = input('free form comment for this file (optional):')
    
        file_uuid = str(uuid.uuid1())
        experiment_uuid =  df_selected.loc[df_selected.index[-1],'uuid_experiment']
        
        filename = '{}_{}_{}.tiff'.format(''.join(filter(str.isalnum, 
                                                         str(df_selected
                                                             .loc[df_selected.index[-1],'date']))), # date 
                                         
                                          df_selected
                                          .loc[df_selected.index[-1],'name']
                                          .split(' ')[-1],                                          # experimenter surname
                                         
                                          file_uuid[:8],                                            # short uuid, file level
                                         )
            
        return get_git_hash(), file_uuid, experiment_uuid, free_form_comment, filename


def analysis_meta(df_selected, entry_type):
    if entry_type=='mask':
        if len(df_selected.index) != 1:
            return 'select one raw file to associate with, repeat the selection'
            
        else:
            def get_git_hash():
                return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        
            free_form_comment = input('free form comment for this mask (optional):')
        
            analysis_uuid = str(uuid.uuid1())
            
            filename = '{}_{}_{}.{}'.format(''.join(filter(str.isalnum, 
                                                             str(df_selected
                                                                 .loc[df_selected.index[-1],'date']))), # date 
                                             
                                              df_selected
                                              .loc[df_selected.index[-1],'name']
                                              .split(' ')[-1],                                          # experimenter surname
                                             
                                              analysis_uuid[:8],                                        # short uuid, analysis level
                                              'npy',
                                             )
                
            return get_git_hash(), analysis_uuid, free_form_comment, filename
    else:
        return 'unknown analysis type'

def read_h5_meta(f):
    # f is h5 file

    try:
        t = f['MetaData']['temperature'].attrs['temperature'].decode('utf-8')
        np_t = np.array([float(value[1:]) for value in t.split('\na') if value.startswith('F')])
        t = str(tuple(np.percentile(np_t, [10, 50, 90]))) 
    except: # (KeyError, IndexError)
        t = 'no temperature data'
    
    conditions = {'name': '',
                     'date': '',
                     'dye': '',
                     'species': '',
                     'strain': '',
                     'temperature': '',
                     'medium': '',
                     'chemical': '',
                     'induction': '',
                     'comment': '',
                     'git_hash': '',
                     'uuid_experiment': ''}


    conditions['name'] = f['MetaData/Sample'].attrs['User'].decode('utf-8')
    date = f['MetaData'].attrs['StartTime']
    conditions['date'] = str(datetime.fromtimestamp(date).date())

    
    conditions['dye'] = f['MetaData/Sample'].attrs['Dye'].decode('utf-8')
    conditions['species'] = f['MetaData/Sample'].attrs['Species'].decode('utf-8')
    conditions['strain'] = f['MetaData/Sample'].attrs['Strain'].decode('utf-8')
    conditions['temperature'] = t
    conditions['medium'] = f['MetaData/Sample'].attrs['Media'].decode('utf-8')
    conditions['chemical'] = f['MetaData/Sample'].attrs['Chemical'].decode('utf-8')
    conditions['induction'] = ''
    comment = f['MetaData/Sample'].attrs['Notes']
    conditions['comment'] = comment.decode('utf-8') if type(comment)!=h5py._hl.base.Empty else '' # experiment level comment
    conditions['git_hash'] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    conditions['uuid_experiment'] = str(uuid.uuid1())
    
    return conditions

def h5_file_uuids(df_selected):

    def get_git_hash():
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

    file_uuid = str(uuid.uuid1())
    experiment_uuid =  df_selected.loc[df_selected.index[-1],'uuid_experiment']
          
    return get_git_hash(), file_uuid, experiment_uuid


