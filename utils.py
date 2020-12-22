import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def load_nifti(filename, with_affine=False):
    """
    load image from NIFTI file
    Args:
        filename (str): filename of NIFTI file
        with_affine (boolean): if True, returns affine parameters
    Returns:
        data (np.ndarray): image data
    """
    img = nib.load(filename)
    data = img.get_fdata()
    data = np.copy(data, order="C")
    if with_affine:
        return data, img.affine
    return data


def plot_img_and_mask(img, mask):
    """
    plot image and the corresponding mask
    # TODO write Docstring here
    Args:
        img: 
    """
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()