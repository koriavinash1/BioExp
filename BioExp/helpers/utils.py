from glob import glob
import numpy as np
import SimpleITK as sitk
import PIL
import tensorflow as tf
import os
import tempfile
from keras.utils import np_utils


def normalize_scheme(slicennot):
    """
        -normalizes each slice, excluding gt
        -subtracts mean and div by std dev for each slice
        -clips top and bottom one percent of pixel intensities
    """
    normed_slices = np.zeros(( 4,155, 240, 240))
    for slicenix in range(4):
        normed_slices[slicenix] = slicennot[slicenix]
        for mode_ix in range(155):
            normed_slices[slicenix][mode_ix] = _normalize(slicennot[slicenix][mode_ix])

    return normed_slices    


def _normalize(slice):

    b = np.percentile(slice, 99)
    t = np.percentile(slice, 1)
    slice = np.clip(slice, t, b)
    image_nonzero = slice[np.nonzero(slice)]

    if np.std(slice)==0 or np.std(image_nonzero) == 0:
        return slice
    else:
        # tmp[tmp==tmp.min()]=-9
        tmp = (slice - np.min(image_nonzero))/(np.max(image_nonzero) - np.min(image_nonzero))
        return tmp

def load_vol_brats( rootpath, 
                    slicen = -1,
                    pad = None):

    """
        loads volume if exists

        rootpath : patient data root path
        slicen : sice which needs to ne loaded
        pad : number of pixels to be padded
                in X, Y direction
    """
    flair = glob( rootpath + '/*_flair.nii.gz')
    t2 = glob( rootpath + '/*_t2.nii.gz')
    gt = glob( rootpath + '/*_seg.nii.gz')
    t1s = glob( rootpath + '/*_t1.nii.gz')
    t1c = glob( rootpath + '/*_t1ce.nii.gz')
    
    t1=[scan for scan in t1s if scan not in t1c]

    if (len(flair)+len(t2)+len(gt)+len(t1)+len(t1c)) < 5:
        print("there is a problem here!!! the problem lies in this patient :")

    scans_test = [flair[0], t1[0], t1c[0], t2[0]]
    test_im = [sitk.GetArrayFromImage(sitk.ReadImage(scans_test[i])) for i in range(len(scans_test))]
    test_im = np.array(test_im).astype(np.float32)
    test_image = normalize_scheme(test_im)
    test_image = test_image.swapaxes(0,1)
    test_image = np.transpose(test_image,(0,2,3,1))

    if pad:
        npad = ((0,0), (pad, pad), (pad, pad), (0, 0))

        test_image = np.pad(test_image, pad_width=npad, mode='constant', constant_values=test_image.min())

    if not slicen == -1:
        test_image = np.array(test_image[slicen])


    try:
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gt[0]))
        gt[gt == 4] = 3

        if pad:
            npad  = ((0,0), (pad, pad), (pad, pad))
            gt    = np.pad(gt, pad_width=npad, mode='constant', constant_values=0)
        if not slicen == -1:
            gt  = np.array(gt[slicen])
        return test_image, gt


    except:
        return test_image


def load_vol( t1path, t2path, t1cepath, flairpath, 
                segpath = None,
                slicen = -1, 
                pad = None):
    """
        loads volume if exists

        rootpath : patient data root path
        slicen : sice which needs to ne loaded
        pad : number of pixels to be padded
                in X, Y direction
    """

    test_im = []
    for pth in [t1path, t2path, t1cepath, flairpath]:
        try: 
            test_im.append(sitk.GetArrayFromImage(sitk.ReadImage(pth)))
        except:
            raise ValueError ("Path doesn't exist: {}".format(pth))

    test_im = np.array(test_im).astype(np.float32)
    gt = test_im[-2]
    gt[gt == 4] = 3

    test_image = normalize_scheme(test_im)
    test_image = test_image.swapaxes(0,1)
    test_image = np.transpose(test_image,(0,2,3,1))

    if pad:
        npad = ((pad, pad), (pad, pad), (0, 0), (0, 0))
        test_image = np.pad(test_image, pad_width=npad, mode='constant', constant_values=0)
        
    if not slicen == -1:
        test_image = np.array(test_image[slicen])

    if segpath:
        gt = sitk.GetArrayFromImage(sitk.ReadImage(segpath))
        gt[gt == 4] = 3

        if pad:
            npad  = ((pad, pad), (pad, pad), (0, 0))
            gt    = np.pad(gt, pad_width=npad, mode='constant', constant_values=0)
        if not slicen == -1:
            gt  = np.array(gt[slicen])
        return test_image, gt

    return test_image


def load_file( rgbpath, maskpath=None):
    """
        loads rgb image

        rgbpath: rgb image path
        maskpath: segmentation path if exists
    """
    rgb  = PIL.Image.open(rgbpath).convert('RGB')

    """TODO: specific preprocessing"""

    try: 
        mask = PIL.Image.open(maskpath).convert('L')
    except:
        return rgb

    return rgb, mask


def predict_volume_brats(model, test_image, show=False):
    """
        Predictions for brats dataset
        involves renaming of classes

        model: keras model
        test_image: image (H x W x C)
        show: bool, to display prediction
    """
    
    test_image = test_image[None, ...]
    
    prediction = model.predict(test_image, batch_size=1) 
    prediction_probs = prediction.copy()
    prediction = np.argmax(prediction, axis=-1)
    prediction=prediction.astype(np.uint8)


    prediction[prediction==3]=4
    
    if show:
        plt.subplot(1,2,1)
        plt.imshow(test_image[:,:,3])
        plt.subplot(1,2,2)
        plt.imshow(prediction[0])
        plt.show()
    
    
    return np.array(prediction), np.array(prediction_probs)


def load_numpy_slice(img_path, mask_path=None, seq='all', pad = 0):
    """
    """
    seq_map = {'flair': 0, 't1': 1, 't2': 3, 't1c':2, 'all':[0, 1, 2, 3]}

    seq = seq_map[seq] 
    img = np.load(img_path)
    img = img[:,:,seq]
    if len(img.shape) == 2:
        img = img[..., None]
        if pad:
            npad = ((pad, pad), (pad, pad), (0, 0))
            img  = np.pad(img, pad_width=npad, mode='constant', constant_values=0)
        return img

    if mask_path:
        mask = np.load(mask_path)[...,None]
        if pad:
            npad = ((pad, pad), (pad, pad), (0, 0))
            img  = np.pad(img, pad_width=npad, mode='constant', constant_values=0)
            mask  = np.pad(mask, pad_width=npad, mode='constant', constant_values=0)
        return img, mask



def load_images(img_path, normalize=True, zscore=False, mask=True):
    """
    """
    if not mask:
        img = np.array(PIL.Image.open(img_path).convert('RGB'))
    else:
        img = np.array(PIL.Image.open(img_path).convert('L'))
    
    if normalize:
        img = (img - np.min(img))/(np.max(img) - np.min(img))

    if zscore:
        img = (img - np.mean(img))/ np.std(img)
    
    return img

def apply_modifications_custom(model, custom_objects=None):
    """Applies modifications to the model layers to create a new Graph. For example, simply changing
    `model.layers[idx].activation = new activation` does not change the graph. The entire graph needs to be updated
    with modified inbound and outbound tensors because of change in layer building function.
    Args:
        model: The `keras.models.Model` instance.
    Returns:
        The modified model with changes applied. Does not mutate the original `model`.
    """
    # The strategy is to save the modified model and load it back. This is done because setting the activation
    # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
    # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
    # multiple inbound and outbound nodes are allowed with the Graph API.
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    finally:
        os.remove(model_path)

def one_hot(tensor, n_classes):
    return(np_utils.to_categorical(tensor, num_classes=n_classes))

