import xml.etree.ElementTree as ET
import os
import pandas as pd

def get_xml(conditions):

    # assembling the xml tree
    root = ET.Element('root')
    
    # assigning the UUID at experiment level
    UUID = ET.SubElement(root, 'UUID')
    UUID.text = conditions['uuid_experiment']
    
    # logging the script version used for xml generation
    ScriptVersion = ET.SubElement(root, 'ScriptVersion')
    ScriptVersion.text = conditions['git_hash']
    
    
    # group/experimenter info, ASCII according to TIFF specifications
    ExperimenterGroup = ET.SubElement(root, 'ExperimenterGroup')
    ExperimenterGroup.text = 'smDyCe lab, Jarno P. Makela, Aalto University, jarno.p.makela@aalto.fi '
    
    Experimenter = ET.SubElement(root, 'Experimenter')
    Experimenter.text = ' {} '.format(conditions['name'])
    
    # experimental conditions
    Experiment = ET.SubElement(root, 'Experiment')
    
    for subelement in list(conditions)[1:10]:
        globals()[subelement.capitalize()] = ET.SubElement(Experiment, subelement.capitalize())
        globals()[subelement.capitalize()].text = ' {} '.format(conditions[subelement])
    
    # writing experiment meta to file
    ET.indent(root, '  ')
    r_xml = ET.ElementTree(root)
    
    out_path = os.getenv('SMPP_DATA_DIR')+'/meta_xml/'
    filename = '{}_{}_{}.xml'.format(''.join(filter(str.isalnum, str(conditions['date']))),  # date
                                     conditions['name'].split(' ')[-1],                      # experimenter surname
                                     conditions['uuid_experiment'][:8],                      # short uuid, experiment level
                                     )
    
    xml_path = out_path+filename
    r_xml.write(xml_path, encoding='utf-8', xml_declaration=True)
    print('metadata xml created!')
    #print(ET.tostring(root).decode('ascii'))

    return xml_path
    
def get_xml_from_h5(h5file, conditions, exp_name):

    # assembling the xml tree
    root = ET.Element('root')
    
    # assigning the UUID at experiment level
    UUID = ET.SubElement(root, 'UUID')
    UUID.text = conditions['uuid_experiment']
    
    # logging the script version used for xml generation
    ScriptVersion = ET.SubElement(root, 'ScriptVersion')
    ScriptVersion.text = conditions['git_hash']
    
    
    # group/experimenter info, ASCII according to TIFF specifications
    ExperimenterGroup = ET.SubElement(root, 'ExperimenterGroup')
    ExperimenterGroup.text = ' smDyCe lab, Jarno P. Makela, Aalto University, jarno.p.makela@aalto.fi '
    
    Experimenter = ET.SubElement(root, 'Experimenter')
    Experimenter.text = ' {} '.format(conditions['name'])
    
    # experimental conditions
    Experiment = ET.SubElement(root, 'Experiment')
    
    for subelement in list(conditions)[1:10]:
        globals()[subelement.capitalize()] = ET.SubElement(Experiment, subelement.capitalize())
        globals()[subelement.capitalize()].text = ' {} '.format(conditions[subelement])

    # experimental conditions
    Instrument = ET.SubElement(root, 'Instrument')
    
    Pixel_micron = ET.SubElement(Instrument, 'Pixel_micron')
    Pixel_micron.text = h5file['MetaData/voxelsize'].attrs['x'].astype('str')#.decode('ascii')

    Camera = ET.SubElement(Instrument, 'Camera')

    CycleTime = ET.SubElement(Camera, 'CycleTime')
    CycleTime.text = h5file['MetaData/Camera'].attrs['CycleTime'].astype('str')#.decode('ascii')

    IntegrationTime = ET.SubElement(Camera, 'IntegrationTime')
    IntegrationTime.text = h5file['MetaData/Camera'].attrs['IntegrationTime'].astype('str')#.decode('ascii')
    
    # writing experiment meta to file
    ET.indent(root, '  ')
    r_xml = ET.ElementTree(root)
    
    out_path = os.getenv('SMPP_DATA_DIR')+'/meta_xml/'
    # out_path = 'utils/temp/auxiliary/' # debug
    filename = '{}_{}.xml'.format(exp_name,  
                                     conditions['uuid_experiment'][:8], # short uuid, experiment level
                                     )
    
    xml_path = out_path+filename
    r_xml.write(xml_path, encoding='utf-8', xml_declaration=True)
    print('metadata xml created!')
    # print(ET.tostring(root).decode('ascii'))

    return xml_path



def get_exps_meta_from_xml(list_mask_ids):

    db = pd.read_csv(os.getenv('SMPP_DATABASE'), sep = '\t', dtype = 'str')
    exp_ids = db[db['uuid_analysis'].isin(list_mask_ids)]['uuid_experiment'].unique()
    paths   = db[db['uuid_experiment'].isin(exp_ids)&db['uuid_fov'].isnull()]['path'].unique()
    # print(paths)
    
    pixel_list = []
    dT_list = []
    
    for path in paths:
        if path[:4]=='data':
            path=os.getenv('SMPP_DATA_DIR')+'/'.join(path.split('/')[1:])
            # print(path)

        
        xml = ET.parse(path)
        root = xml.getroot()
        
        pixel = float(root[5][0].text)
        if pixel>1:
            pixel = pixel/1000 # for old h5 files where voxelsize is in nm
        pixel_list.append(pixel)
        # print(pixel_list)
        
        dT = float(root[5][1][0].text)
        dT_list.append(dT)
        # print(dT_list)


        

    assert all(i==pixel_list[0] for i in pixel_list), 'noooooooooo!!! pixel size is not the same!!!'
    assert all(i==dT_list[0] for i in dT_list), 'noooooooooo!!! IntegrationTime is not the same!!!'
    
    return (pixel_list[0], dT_list[0])

    


