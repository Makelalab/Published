import tifffile
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib
from PIL import Image
import boto3
import os



def create_array(directory, bright_choice, fluo_choice):
    bright = [np.mean(tifffile.imread(Path(directory)/bright_choice), axis = 0)
          .astype('uint16')[np.newaxis,:]]
    all_fluo = [tifffile.imread(Path(directory)/i) for i in sorted(fluo_choice)]


    arr = np.concatenate(bright+all_fluo)
    #print(bright[0].shape)
    #print(all_fluo[0].shape)
    #print(arr.shape)
    
    plt.imshow(arr[0,:,:])
    plt.title('Brightfield projection')
    plt.show()
    
    plt.imshow(np.mean(arr[1:,:,:], axis = 0))
    plt.title('Fluorescence channel projection')
    plt.show()

    plt.imshow(np.std(arr[1:,:,:], axis = 0))
    plt.title('SD of fluorescence signal')
    plt.show()

    return arr


def show_mask(directory, mask_choice, imshow=True):
    
    if mask_choice[-4:] == '.png':
        #print('png')
        m = Image.open(Path(directory)/mask_choice)  
        m = np.array(m)
        shfld = np.arange(10, np.max(m)+10)
        np.random.shuffle(shfld)

        mm = np.where(m == 0, 0, m)
        for i in range(1, np.max(m)):
            mm = np.where(m == i, shfld[i], mm)

        if imshow:
            plt.imshow(mm, cmap = 'Blues')
            plt.show()    
    
    if mask_choice[-4:] == '.npy':
        #print('npy')
        m_arr = np.load(Path(directory)/mask_choice, allow_pickle=True)
        m = m_arr.item()
        m = m['masks']
        shfld = np.arange(10, np.max(m)+10)
        np.random.shuffle(shfld)

        mm = np.where(m == 0, 0, m)
        for i in range(1, np.max(m)):
            mm = np.where(m == i, shfld[i], mm)

        if imshow:
            plt.imshow(mm, cmap = 'Blues')
            plt.show()    
            
    return m


def write_tiff(path, array, uuid_f, uuid_e, free_text, name):
    
    time_interval = 0.010475 # this is hard-coded for now
    
    tifffile.imwrite(path+name,
                 array,
                 imagej=True,
                 resolution=(1./0.106, 1./0.106), # also hard-coded
                 metadata={
                     'unit':'um',
                     'finterval':time_interval,
                     'fps':1/time_interval,
                     'axes':'TYX',
                     'ImageDescription': '<UUID_file> {} </UUID_file><UUID_experiment> {} </UUID_experiment><Comment> {} </Comment>'.format(uuid_f, uuid_e, free_text),
                     'Labels':[f't {i*time_interval}' for i in range(array.shape[0])]
                 }
                 )
    print('All is fine, tiff is ready.')

def upload_img_allas(directory, file_name):
    # s3_resource.create_bucket(Bucket="smt_0")
    # for bucket in s3_resource.buckets.all():
    #     print(bucket.name)
    
    s3_resource = boto3.resource('s3', endpoint_url=os.getenv('SMPP_S3_ENDPOINT'), config={})
    my_bucket = s3_resource.Bucket(os.getenv('SMPP_S3_BUCKET'))

    # for my_bucket_object in my_bucket.objects.all():
    #     print(my_bucket_object.key)

    obj = s3_resource.Object(os.getenv('SMPP_S3_BUCKET'), file_name)
    obj.upload_file(directory+file_name)

    # s3_resource.Object('smt_0', file_name).download_file(directory+'anyname.tiff') 

    print('File uploaded to Allas, bucket: {}, object: {}'.format(my_bucket.name, obj.key))

    return my_bucket.name, obj.key


def download_img_allas(s3_bucket, s3_object):

    

    # https://stackoverflow.com/questions/38905291/is-it-possible-to-get-the-contents-of-an-s3-file-without-downloading-it-using-bo
    # s3 = boto3.resource('s3')
    # print s3.Object('mybucket', 'beer').get()['Body'].read()
 
    s3_resource = boto3.resource('s3', endpoint_url=os.getenv('SMPP_S3_ENDPOINT'))
    # for bucket in s3_resource.buckets.all():
    #     print(bucket.name)
        
    # my_bucket = s3_resource.Bucket('smt_0')
    # for my_bucket_object in my_bucket.objects.all():
    #     print(my_bucket_object.key)

    tiff = s3_resource.Object(s3_bucket, s3_object).get()['Body'].read()

    # s3_resource.Object('smt_0', file_name).download_file(directory+'anyname.tiff') 

    return tiff


def h5_upload_img_allas(directory, file_name):
    # s3_resource.create_bucket(Bucket="smt_0")
    # for bucket in s3_resource.buckets.all():
    #     print(bucket.name)
    
    s3_resource = boto3.resource('s3', endpoint_url=os.getenv('SMPP_S3_ENDPOINT'))
    my_bucket = s3_resource.Bucket(os.getenv('SMPP_S3_BUCKET'))
    # my_bucket = s3_resource.Bucket('test_smt2')

    # for my_bucket_object in my_bucket.objects.all():
    #     print(my_bucket_object.key)

    obj = s3_resource.Object(os.getenv('SMPP_S3_BUCKET'), file_name)
    obj.upload_file(Path(directory)/file_name)

    # s3_resource.Object('smt_0', file_name).download_file(directory+'anyname.tiff') 

    # print('THIS IS TEST BUCKET File uploaded to Allas, bucket: {}, object: {}'.format(my_bucket.name, obj.key))
    print('File uploaded to Allas, bucket: {}, object: {}'.format(my_bucket.name, obj.key))
    
    return my_bucket.name, obj.key


def fluo_as_array(image_2ch_fxy):
    
    # This converts 2d fluorescence image (2ch projection) into an array of (x,y,value) rows
    
    fluo_2ch_proj = np.mean(image_2ch_fxy, axis=0)
    f2ch_as_array = np.array([(i, j, fluo_2ch_proj[i, j]) for i in range(fluo_2ch_proj.shape[0]) for j in range(fluo_2ch_proj.shape[1])])
    print(f2ch_as_array.shape)
    f2ch_as_array_df = pd.DataFrame(f2ch_as_array, columns=['x', 'y', 'value'])

    return f2ch_as_array_df
    

def coords_to_image(data: pd.DataFrame, value: str) -> np.ndarray:
    
    # convert a df with x, y, value columns to np.array of shape x, y
    # display(data) # debug
    
    x, y, value = data['x_rounded'], data['y_rounded'], data[value]
    
    
    x, y = np.asarray(x, dtype=int), np.asarray(y, dtype=int)
    arr = np.full((x.max() + 10, y.max() + 10), np.nan)
    arr[x, y] = np.asarray(value)
    return arr
