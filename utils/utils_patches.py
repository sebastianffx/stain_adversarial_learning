import numbers
from skimage.util.shape import view_as_windows
import glob
from sklearn.preprocessing import OneHotEncoder
import random

from glob import iglob
import fnmatch
from numpy.lib.stride_tricks import as_strided
from matplotlib.pyplot import imread
import csv
import matplotlib.pyplot as plt
from PIL import * 
from config import *
import numpy as np

import time


##Building the generators for balanced negative-positive batches
class simplePatchGenerator(object):
    def __init__(self, input_dir, batch_size, img_shape = (63,63,3), augmentation_fn=None):
        print("Building generator for patches in " + input_dir)
        # Params
        self.input_dir = input_dir  # path to patches in glob format
        self.batch_size = batch_size  # number of patches per batch
        self.augmentation_fn = augmentation_fn  # augmentation function
        self.img_shape = img_shape
        self.list_positive = glob.glob(input_dir + 'mitosis/*.png')
        self.list_negative = glob.glob(input_dir + 'non_mitosis/*.png')
        self.encoder = OneHotEncoder(sparse=False)
        

        self.n_samples = len(self.list_negative +self.list_positive)
        self.n_batches = self.n_samples // self.batch_size
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        # Provide length in number of batches
        return self.n_batches

    def next(self):
        # Build a mini-batch
        paths_negatives = random.sample(self.list_negative,self.batch_size//2) #self.df.loc[self.df['label'] == 0, :].sample(self.batch_size//2, replace = True)
        paths_positives = random.sample(self.list_positive,self.batch_size//2) #self.df.loc[self.df['label'] == 0, :].sample(self.batch_size//2, replace = True)
        images = []
        labels = np.concatenate((np.zeros((self.batch_size//2,1)), np.ones((self.batch_size//2,1))),axis=0)
        label_idx = 0
        for pathimg in paths_negatives+paths_positives:
            try:
                # Read image path
                # Read data and label
                image = load_image(pathimg)
                # Data augmentation
                if self.augmentation_fn:
                    image = self.augmentation_fn(image)

                # Append
                if image.shape != self.img_shape:
                    images.append(images[-1])
                    labels[label_idx] = labels[label_idx-1]
                else:
                    images.append(image)
                label_idx+=1
            except Exception as e:
                print('Failed reading img {idx}...'.format(idx=pathimg))
                print(e)
            label_idx += 1
            
        batch_x = np.stack(images).astype('float32')
        batch_y = np.stack(labels).astype('float32')

        return batch_x, self.encoder.fit_transform(batch_y)#batch_y


def localize_mitosis(img,pathcsv):
    coord_mitosis = []
    with open(pathcsv, 'rt', encoding='utf8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            coord_mitosis.append(np.array([int(row[0]),int(row[1])]))
    return coord_mitosis

def localize_mitosis_patches(pathcsv,return_img=False):
    coord_mitosis = []
    mitotic_patches = []
    num_case = int(pathcsv.split('/')[-2])
    str_case = pathcsv.split('/')[-2]
    str_hpf  = pathcsv.split('/')[-1].split('.csv')[0]
    with open(pathcsv, 'rt', encoding='utf8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            coord_mitosis.append(np.array([int(row[0]),int(row[1])]))
    if num_case <= 14:
        part = '1'
    if (num_case > 14) and (num_case <= 33) :
        part = '2'    
    if (num_case > 33) :
        part = '3'
    path_img = PREFIX_IMGS + part + '/' + str_case + '/' + str_hpf + '.tif'
    sampl_img = np.array(Image.open(path_img))
    for item in coord_mitosis:
        anchor_x, anchor_y = item[0]-31,  item[1]-31
        if anchor_x < 0:  
            anchor_x = 0
        if anchor_y < 0:
            anchor_y = 0          
        cur_patch =sampl_img[anchor_x:anchor_x+63,anchor_y:anchor_y+63,:] 
        mitotic_patches.append(cur_patch)
    print("Computed coords and patches for " + path_img ) 
    if return_img: 
        return coord_mitosis,mitotic_patches,sampl_img
    else:
        return coord_mitosis,mitotic_patches

def extract_patches(arr, patch_shape=8, extraction_step=1):
    """Extracts patches of any n-dimensional array in place using strides.
    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted
    patch_shape : integer or tuple of length arr.ndim
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.
    extraction_step : integer or tuple of length arr.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    """

    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = [slice(None, None, st) for st in extraction_step]
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))
    patches = as_strided(arr, shape=shape, strides=strides)
    return patches, indexing_strides, patch_indices_shape, slices

def is_white_patch(cur_patch,white_percentage):
    is_white = True
    total_white = float(cur_patch.shape[0] *cur_patch.shape[1] * cur_patch.shape[2] * 255)
    if (cur_patch.sum()/total_white)>white_percentage:
        return is_white
    else:
        return not is_white
def read_open_slide_image(path,level):
    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open(path)
    ds = mr_image.getLevelDownsample(level)
    pixels = mr_image.getUCharPatch(int(1 * ds), int(1 * ds), mr_image.getLevelDimensions(level)[0], mr_image.getLevelDimensions(level)[1], level)
    return pixels

def compute_corner_patches(img_size, stride_x, stride_y):
    array_corners = []
    total_p_x,total_p_y  = 0,0
#    for i in range(0,img_size[0]-5*stride_x,stride_x): #Works for 48x48, 8 stride patches
    stride_x_corner = 0
    stride_y_corner = 0
    for i in range(0,img_size[0]-stride_x_corner*stride_x,stride_x):
        total_p_x = total_p_x +1
        corners_row = []
#        for j in range(0,img_size[1]-5*stride_y, stride_y):
        for j in range(0,img_size[1]-stride_y_corner*stride_y, stride_y):
            if i == 0:
                total_p_y = total_p_y +1
            corners_row.append((i,j))
        array_corners.append(corners_row)
    return np.array(array_corners),total_p_x,total_p_y


#This function is not mine, is from a course from Radboud University Medical Center, super handy!
# Define a function to read images from disk and convert them to xyc format in a desire output range.
def load_image(input_path, range_min=0, range_max=1):
    
    # Read image data (x, y, c) [0, 255]
    image = imread(input_path)
    
    # Convert image to the correct range
    image = (image / 255) * (range_max - range_min) + range_min 

    return image

# Define a function to plot a batch or list of image patches in a grid
def plot_image(images, images_per_row=8):
    fig, axs = plt.subplots(int(np.ceil(len(images)/images_per_row)), images_per_row)
    c = 0
    for ax_row in axs:
        for ax in ax_row:
            if c < len(images):
                ax.imshow(images[c])
            ax.axis('off')            
            c += 1
    plt.show()

    



def save_heatmap(fname,cur_slide, coordinates, boxes_to_draw):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(20,20)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(cur_slide) 

    for item in coordinates[0:boxes_to_draw]:
        # Create a Rectangle patch
        rect = patches.Rectangle((item[1],item[0]),10,10,linewidth=2,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    #fig.savefig(, 300)
    fig.savefig(fname, bbox_inches='tight', pad_inches=0)
    return


def compute_probs_sliding_window_v2(model_mitosis,ex_img):
    probs_row = []
    stride_x,stride_y = 63,63
    start = time.time()
    for i in range(ex_img.shape[0]-stride_x): #This is a nice use case of list comprehension, for GPU mem limists is not possible to use only one line ranging also in the rows dim
        cur_batch_row = np.squeeze(np.array([[ex_img[i:i+63,j:j+63,:] for j in range(ex_img.shape[1]-stride_y)]]))        
        probs_row.append(model_mitosis.predict_proba(cur_batch_row,batch_size=cur_batch_row.shape[0]))
    end = time.time()
    print(str(i)+" iterations - Time elapsed on gpu + comprehension :" + str(end - start))
    return (np.array(probs_row))




# Sliding window CNN approach using list-comprehensions to speed up a little bit
def compute_probs_sliding_window_v1(model_mitosis,ex_img):
    kk_r = extract_patches(ex_img[:,:,0], patch_shape=63, extraction_step=1)[0]
    kk_g = extract_patches(ex_img[:,:,1], patch_shape=63, extraction_step=1)[0]
    kk_b = extract_patches(ex_img[:,:,2], patch_shape=63, extraction_step=1)[0]
    probs_row = []
    for i in range(kk_r.shape[0]): #This is a nice use case of list comprehension, for GPU mem limits is not possible to use only one line, i.e., ranging also in the rows dim
        cur_thing_r = np.squeeze(np.array([[kk_r[i,j,:,:] for j in range(kk_r.shape[1])]]))
        cur_thing_g = np.squeeze(np.array([[kk_g[i,j,:,:] for j in range(kk_r.shape[1])]]))
        cur_thing_b = np.squeeze(np.array([[kk_b[i,j,:,:] for j in range(kk_r.shape[1])]]))
        start = time.time()
        cur_thing = np.stack((cur_thing_r, cur_thing_g, cur_thing_b),axis=-1)
        #start = time.time()
        probs_row.append(model_mitosis.predict_proba(cur_thing,batch_size=cur_thing.shape[0]))
        #probs_row.append(model_mitosis.predict_proba(np.concatenate((cur_thing_sss,cur_thing_sss),axis=0)))
        
        #print("Time elapsed on gpu :" + str(end - start))
        if i%100== 0:
            print("Computing probs for row " + str(i))
        end = time.time()
    return(np.array(probs_row))

