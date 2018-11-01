import numbers
from skimage.util.shape import view_as_windows
from glob import iglob
import fnmatch
from numpy.lib.stride_tricks import as_strided

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
