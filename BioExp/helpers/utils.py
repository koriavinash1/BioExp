from glob import glob
import numpy as np
import SimpleITK as sitk
import PIL


def normalize_scheme(slicennot):
    """
        normalizes each slice, excluding gt
        subtracts mean and div by std dev for each slice
        clips top and bottom one percent of pixel intensities
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
        tmp= (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        tmp[tmp==tmp.min()]=-9
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
        npad = ((pad, pad), (pad, pad), (0, 0), (0, 0))
        test_image = np.pad(test_image, pad_width=npad, mode='constant', constant_values=0)

    if not slicen == -1:
        test_image = np.array(test_image[slicen])


    try:
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gt[0]))
        gt[gt == 4] = 3

        if pad:
            npad  = ((pad, pad), (pad, pad), (0, 0))
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
