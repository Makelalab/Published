from ipywidgets import Button, Layout, interactive, fixed
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, clear_output

def experiment_widgets():

    name_w = widgets.Dropdown(
        options=['Jarno Makela', 
                 'Ada Pajari', 
                 'Chris Hayes',
                 'Aakeel Wagay',
                 'Dan Noel',
                 'Taras Redchuk',
                 'Eveliny Nery',
                 'Noora Roos',
                 'other / ours / value',
                ],
        value='other / ours / value',
        description='Name:',
        disabled=False,
        )
    
    date_w = widgets.DatePicker(
        description='Pick a Date',
        disabled=False
        )
    
    dye_w = widgets.Dropdown(
        options=['JFX554', 
                 'JFX650', 
                 'JF503',
                 'TMR',
                 'DAPI',
                 'multiple',
                 'other / ours / value',
                ],
        value='other / ours / value',
        description='Dye:',
        disabled=False,
        )
    
    species_w = widgets.Dropdown(
        options = ['Escherichia coli',
                   'Bacillus subtilis',
                   'Geobacillus stearothermophilus',
                   'Lactococcus lactis',
                   'Psychrobacter cryohalolentis',
                   'Pseudoalteromonas translucida',
                   'Thermus thermophilus',
                   'other / ours / value',
                  ],
        value='other / ours / value',
        description='Species:',
        disabled=False,
        )
    
    strain_w = widgets.Dropdown(
        options = ['CJW6340',
                   'EL605',
                   'EL689',
                   'other / ours / value',
                  ],
        value='other / ours / value',
        description='Strain:',
        disabled=False,
        )
    
    temperature_w = widgets.Dropdown(
        options = ['heat shock',
                   'cold shock',
                   'complicated',
                   'other / ours / value',
                  ],
        value='other / ours / value',
        description='Temperature:',
        disabled=False,
        )
    
    medium_w = widgets.Dropdown(
        options = ['M9glyCAAT',
                   'LB',
                   'other / ours / value',
                  ],
        value='other / ours / value',
        description='Medium:',
        disabled=False,
        )
    
    chemical_w = widgets.Dropdown(
        options = ['rifampicin',
                   'chloramphenicol',
                   'ampicillin',
                   'chloroquine',
                   'ciprofloxacin',
                   'other / ours / value',
                  ],
        value='other / ours / value',
        description='Chemical:',
        disabled=False,
        )
    
    induction_w = widgets.Checkbox(
        value=False,
        description='Induction',
        disabled=False,
        indent=True
    )
    
    comment_w = widgets.Textarea(
        value='',
        placeholder='enter free form text',
        description='Comment:',
        disabled=False
    )
    
    
    widgets_list = [name_w,
                    date_w,
                    dye_w, 
                    species_w, 
                    strain_w, 
                    temperature_w, 
                    medium_w, 
                    chemical_w, 
                    induction_w,
                    comment_w,
                   ]
    
    conditions_list = ['name',
                       'date', 
                       'dye', 
                       'species', 
                       'strain', 
                       'temperature', 
                       'medium', 
                       'chemical', 
                       'induction',
                       'comment',
                   ]
    
    box = widgets.VBox(children = widgets_list)

    return (conditions_list, widgets_list, box)

def im_file_chooser(inp_path):
    tif = []
    
    for fpath in Path(inp_path).glob('*.tif'):
        filename = fpath.name
        tif.append(filename)

    fluo_chooser_w = widgets.SelectMultiple(
        options=tif,
        value=[tif[0]],
        #rows=10,
        description='Fluorescence:',
        disabled=False,
        layout=Layout(width='75%', height='180px')
    )

    bf_chooser_w = widgets.Select(
        options=tif,
        value=tif[0],
        #rows=10,
        description='Brightfield:',
        disabled=False,
        layout=Layout(width='75%', height='180px')
    )

    fbox = widgets.VBox(children = [fluo_chooser_w, bf_chooser_w])
    
    return fbox


def im_t_file_chooser(inp_path):
    csv_fs = []
    
    for fpath in Path(inp_path).glob('*.csv'):
        csv_fs.append(fpath.name)

    fluo_chooser_w = widgets.SelectMultiple(
        options=csv_fs,
        value=[csv_fs[0]],
        #rows=10,
        description='Tracking files:',
        disabled=False,
        layout=Layout(width='75%', height='180px')
    )
  
    return fluo_chooser_w


def mask_chooser(inp_path):
    masks = []
    
    for fpath in Path(inp_path).glob('*.npy'):
        filename = fpath.name
        masks.append(filename)

    for fpath in Path(inp_path).glob('*.png'):
        filename = fpath.name
        masks.append(filename)

    mask_chooser_w = widgets.Select(
        options=masks,
        value=masks[0],
        #rows=10,
        description='Mask files:',
        disabled=False,
        layout=Layout(width='75%', height='180px')
        )

    return mask_chooser_w


def filter_data(df):
    widgets_list = []

    # type_w = widgets.Dropdown(
    #     options=['Experiments', 
    #              'Raw images', 
    #              'Analysis',
    #              'any',
    #             ],
    #     value='any',
    #     description='Type:',
    #     disabled=False,
    #     )
    # widgets_list.append(type_w)

    date_w = widgets.DatePicker(
        description='Pick a Date',
        disabled=False
        )
    widgets_list.append(date_w)
    
    fields_list = [#'type',
                   'date',
                   'name',
                   'dye', 
                   'species', 
                   'strain', 
                   'temperature', 
                   'medium', 
                   'chemical', 
                   'induction',
                   'exclude',
                   'include',
                   ]
    
    for field in fields_list[1:-2]:
        globals()[field+'_w'] = widgets.Dropdown(
            options=df[field].unique().tolist()+['any'],
            value='any',
            description=field.capitalize()+':',
            disabled=False,
            )
        widgets_list.append(globals()[field+'_w'])
    
    exclude_w = widgets.Textarea(
            value='',
            placeholder="enter indices, like '2, 4, 17'",
            description='Exclude:',
            disabled=False
        )
    widgets_list.append(exclude_w)

    include_w = widgets.Textarea(
            value='',
            placeholder="enter indices, like '2, 4, 17'",
            description='Include:',
            disabled=False
        )
    widgets_list.append(include_w)
    
    box = widgets.VBox(children = widgets_list)

    return (fields_list, widgets_list, box)


def w_tracking_params():   
    style = {'description_width': 'initial'}
    tracking_params_list = [widgets.Text(value='1',
                placeholder='enter integer value',
                description='trackpy threshold:',
                disabled=False,
                style=style,
                ),
    
    widgets.Text(value='130',
                placeholder='enter integer value',
                description='matlab threshold:',
                disabled=False,
                style=style,
                ),

    widgets.Text(value='0.0107',
                placeholder='enter value',
                description='dT:',
                disabled=False,
                style=style,
                ),
    
    widgets.Dropdown(
                options = ['auto',
                           'manual',
                           'no alignment',
                          ],
                value='auto',
                description='align mask',
                disabled=False,
                ),

    widgets.Text(value='42',
                placeholder='enter value',
                description='preview frame',
                disabled=False,
                style=style,
                ),
                           ]

    box = widgets.VBox(children = tracking_params_list)
    
    return(box)
    
def entry_type_widget():

    entry_type_w = widgets.Dropdown(
        options=['Experiments', 
                 'Raw images', 
                 'Analysis',
                 'Everything',
                ],
        value='Everything',
        description="I'd like to see:",
        disabled=False,
        )
    

    return entry_type_w   


def correct_or_delete():
    cells_to_change = widgets.VBox(children = [
            widgets.Dropdown(
            options=['name', 
                     'date', 
                     'dye', 
                     'species', 
                     'strain', 
                     'temperature', 
                     'medium',
                     'chemical', 
                     'induction', 
                     'comment'
                    ],
            
            value='comment',
            description='Column:',
            disabled=False,
            
            ),
    
        widgets.Textarea(
                    value='',
                    placeholder='list (1, 2, 42) \nor range (16 - 256)',
                    description='Row:',
                    disabled=False
                ),
        
        widgets.Textarea(
                    value='',
                    placeholder='type desired \nvalue here',
                    description='Set to:',
                    disabled=False
                ),
        
        widgets.Checkbox(
            value=False,
            description='Delete this (keep \'Column\' intact)',
            disabled=False,
            indent=True,
    
        )
    ])
    return cells_to_change

def manual_reg_widgets(fovs_list):
    
    all_fovs_locs = fovs_list
    locs=all_fovs_locs[0][0] # assuming all FOVs have the same shape
    
    fw = widgets.IntSlider(
        value=0,
        min=0,
        max=len(all_fovs_locs)-1,
        step=1,
        description='FOV number:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=Layout(width='300px'),
    )
    
    
    xw = widgets.IntRangeSlider(
        value=[locs['y'].min()-10, locs['y'].max()+10],
        min=locs['y'].min()-10,
        max=locs['y'].max()+10,
        step=5,
        description='xlim:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=Layout(width='500px'),
    )
    
    
    yw = widgets.IntRangeSlider(
        value=[locs['x'].min()-10, locs['x'].max()+10],
        min=locs['x'].min()-10,
        max=locs['x'].max()+10,
        step=5,
        description='ylim:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=Layout(width='500px'),
    )
    
    xs = widgets.IntSlider(
        value=0,
        min=-30,
        max=30,
        step=1,
        description='x offset:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=Layout(width='500px'),
    )
    
    
    ys = widgets.IntSlider(
        value=0,
        min=-30,
        max=30,
        step=1,
        description='y offset:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=Layout(width='500px'),
    )


    return fw, xw, yw, xs, ys


def fov_selector(df_files, df_masks):

    df_files.dropna(subset='uuid_fov', inplace=True)

    widgets_list = []
    checked_ids = []
    output = widgets.Output()
    
    fov_ids = df_files['uuid_fov'].unique()
    
    def on_checkbox_change(change, fov_id):
        if change['new']:
            checked_ids.append(fov_id)
        else:
            checked_ids.remove(fov_id)
        with output:
            clear_output(wait=True)
            print("Checked FOV IDs:", checked_ids)
    
    for idx in fov_ids:
        raw_in_fov = df_files.where(df_files['uuid_fov'] == idx).dropna(how='all', axis=0)
        mask_in_fov = df_masks.where(df_masks['uuid_fov'] == idx).dropna(how='all', axis=0)
        ch_list = [n[-5:] for n in raw_in_fov['comment'].unique()]
    
        description = f"fov id: {idx[:8]}; {raw_in_fov.shape[0]} channels: {ch_list}; masks: {mask_in_fov.shape[0]}"
        
        checkbox = widgets.Checkbox(
            value=False,
            description=description,
            disabled=False,
            indent=False,
            layout=widgets.Layout(width='75%')
        )
        
        checkbox.observe(lambda change, fov_id=idx: on_checkbox_change(change, fov_id), names='value')
        
        widgets_list.append(checkbox)

    box = widgets.VBox(children=widgets_list)
    return(box, output, checked_ids)



def fov_all_selector(df_files, df_masks):
    #drop the colunms which in their uuid-fov is NA
    df_files.dropna(subset='uuid_fov', inplace=True)
    

    widgets_list = []
    select_all_check_box=widgets.Checkbox(
            value=False,
            description='Select all',
            disabled=False,
            indent=False,
            layout=widgets.Layout(width='75%')
        )
    widgets_list.append(select_all_check_box)
    checked_ids = []
    output = widgets.Output()
    
    fov_ids = df_files['uuid_fov'].unique()
    
    def on_checkbox_change(change, fov_id):
        if change['new']:
            checked_ids.append(fov_id)
        else:
            checked_ids.remove(fov_id)
        with output:
            clear_output(wait=True)
            print("Checked FOV IDs:", checked_ids)

    def select_all(change):
        for section in widgets_list[1:]:
            section.value=widgets_list[0].value
    
    for idx in fov_ids:
        raw_in_fov = df_files.where(df_files['uuid_fov'] == idx).dropna(how='all', axis=0)
        mask_in_fov = df_masks.where(df_masks['uuid_fov'] == idx).dropna(how='all', axis=0)
        ch_list = [n[-5:] for n in raw_in_fov['comment'].unique()]
    
        description = f"fov id: {idx[:8]}; {raw_in_fov.shape[0]} channels: {ch_list}; masks: {mask_in_fov.shape[0]}"
        
        check_box = widgets.Checkbox(
            value=False,
            description=description,
            disabled=False,
            indent=False,
            layout=widgets.Layout(width='75%')
        )
        
        check_box.observe(lambda change, fov_id=idx: on_checkbox_change(change, fov_id), names='value')
        
        widgets_list.append(check_box)
        

    box = widgets.VBox(children=widgets_list)
    widgets_list[0].observe(select_all)
    widgets_list[0].value=True
    return(box, output, checked_ids)



def h5_w_tracking_params():   
    style = {'description_width': 'initial'}
    tracking_params_list = [widgets.Text(value='1',
                placeholder='enter integer value',
                description='trackpy threshold:',
                disabled=False,
                style=style,
                ),
    
    widgets.Text(value='130',
                placeholder='enter integer value',
                description='matlab threshold:',
                disabled=False,
                style=style,
                ),

    widgets.Dropdown(
                options = ['auto',
                           'manual',
                           'no alignment',
                          ],
                value='manual',
                description='align mask',
                disabled=False,
                ),

    widgets.Text(value='42',
                placeholder='enter value',
                description='preview frame',
                disabled=False,
                style=style,
                ),
                           
    widgets.Checkbox(
                value=True,
                description='same m_threshold for all FOVs',
                disabled=False,
                indent=True, 
                ),
                            
    widgets.Checkbox(
                value=True,
                description='analyse all frames',
                disabled=False,
                indent=True, 
                ),

    widgets.Checkbox(
                value=False,
                description='remove outliers (4SD)',
                disabled=False,
                indent=True, 
                ),

    widgets.Checkbox(
                value=True,
                description='allow 1 missing frame',
                disabled=False,
                indent=True, 
                ),
                           ]

    box = widgets.VBox(children = tracking_params_list)
    
    return(box)


def h5_w_multiple_ml_thresholds(list_of_tracking_files, default_value):
    style = {'description_width': 'initial'}
    w_list = []
    for tracking_file in list_of_tracking_files:
        w_list.append(widgets.Text(value=default_value,
                    placeholder='enter value',
                    description=tracking_file[1][1].split('_')[-1],
                    disabled=False,
                    style=style,
                    ),
                    )

    box = widgets.VBox(children = w_list)
    
    return(box)



def m_frame_range_wds(fovs_list):
    
    all_fovs_locs = fovs_list

    
    fw = widgets.IntSlider(
        value=0,
        min=0,
        max=len(all_fovs_locs)-1,
        step=1,
        description='FOV number:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=Layout(width='300px'),
    )
    
    
    frame_range = widgets.IntRangeSlider(
        value=[0, 20025],
        min=0,
        max=20025,
        step=10,
        description='frame range:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=Layout(width='600px'),
    )

    
    return fw, frame_range
