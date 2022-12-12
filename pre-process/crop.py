import nibabel
import numpy as np
from scipy import ndimage
import nibabel as nib
import os

result_list = []
mask1_list = []


def to_cube(img, seg):
    img_array = img.get_fdata()
    seg_array = seg.get_fdata().astype('uint8')
    af = img.affine
    zoom1 = abs(af[2, 2] / af[0, 0])
    zoom2 = abs(128.0 / float(np.shape(img_array)[0]))
    img_array = ndimage.zoom(img_array, [zoom2, zoom2, zoom1 * zoom2], order=3, )
    seg_array = ndimage.zoom(seg_array, [zoom2, zoom2, zoom1 * zoom2], order=0, )
    b = np.array([[1 / zoom2, 0, 0, 0], [0, 1 / zoom2, 0, 0], [0, 0, 1 / (zoom1 * zoom2), 0], [0, 0, 0, 1]])
    af_crop = np.matmul(af, b)
    img_array = np.round(img_array).astype('int16')
    seg_array = seg_array.astype('uint8')
    return nib.Nifti1Image(img_array, af_crop), nib.Nifti1Image(seg_array, af_crop)


img_path = r"D:\verse\external_dataset_processed\align\img"
seg_path = r"D:\verse\external_dataset_processed\align\seg"
f_list = os.listdir(img_path)

single_struct = np.array(([1, 1, 0]), dtype=bool)
for f in f_list:
    print(f"========{os.path.splitext(f)[0]}========")
    img = nib.load(os.path.join(img_path, f))
    seg = nib.load(os.path.join(seg_path, f))
    img_arr = img.get_fdata()
    seg_arr = seg.get_fdata().astype('uint8')
    img_head = img.header
    uniq = np.unique(seg_arr)
    af = img.affine

    for i in uniq:
        if i == 0:
            continue
        seg_crop = np.where(seg_arr == i, 1, 0)
        img_crop = img_arr
        mask_2 = seg_crop.sum(axis=0, keepdims=True).sum(axis=1, keepdims=True) != 0
        mask_2 = np.squeeze(mask_2)
        mask_2_range = np.where(mask_2 == True)
        mask_1 = seg_crop.sum(axis=0, keepdims=True).sum(axis=2, keepdims=True) != 0
        mask_1 = np.squeeze(mask_1)
        mask_1_range = np.where(mask_1 == True)
        mask_0 = seg_crop.sum(axis=1, keepdims=True).sum(axis=2, keepdims=True) != 0
        mask_0 = np.squeeze(mask_0)
        mask_0_range = np.where(mask_0 == True)

        mask_2_l = np.max(mask_2_range) - np.min(mask_2_range)
        mask_1_l = np.max(mask_1_range) - np.min(mask_1_range)
        mask_0_l = np.max(mask_0_range) - np.min(mask_0_range)
        l_diff = mask_0_l - mask_1_l

        if l_diff > 0:
            if l_diff % 2 == 1:
                mask_1 = ndimage.binary_dilation(mask_1, structure=single_struct, iterations=1)
                l_diff -= 1
            if l_diff != 0:
                iter = l_diff // 2
                mask_1 = ndimage.binary_dilation(mask_1, iterations=iter)

        elif l_diff < 0:
            l_diff = abs(l_diff)
            if l_diff % 2 == 1:
                mask_0 = ndimage.binary_dilation(mask_0, structure=single_struct, iterations=1)
                l_diff -= 1
            if l_diff != 0:
                iter = l_diff // 2
                mask_0 = ndimage.binary_dilation(mask_0, iterations=iter)

        mask_0_range = np.where(mask_0 == True)
        mask_1_range = np.where(mask_1 == True)
        mask_2_range = np.where(mask_2 == True)
        mask_1_l = np.max(mask_1_range) - np.min(mask_1_range)
        mask_0_l = np.max(mask_0_range) - np.min(mask_0_range)
        mask_2_l = np.max(mask_2_range) - np.min(mask_2_range)
        b = np.array([[1, 0, 0, np.min(mask_0_range)], [0, 1, 0, np.min(mask_1_range)], [0, 0, 1, np.min(mask_2_range)],
                      [0, 0, 0, 1]])

        seg_crop = seg_crop[mask_0, :, :]
        seg_crop = seg_crop[:, mask_1, :]
        seg_crop = seg_crop[:, :, mask_2]
        img_crop = img_crop[mask_0, :, :]
        img_crop = img_crop[:, mask_1, :]
        img_crop = img_crop[:, :, mask_2]

        af_crop = np.matmul(af, b)
        img_crop, seg_crop = to_cube(nibabel.Nifti1Image(img_crop, af_crop), nibabel.Nifti1Image(seg_crop, af_crop))
        # print(img_crop)
        # print(seg_crop)
        img_crop.to_filename(
            rf"D:\verse\external_dataset_processed\crop\img\{os.path.splitext(os.path.splitext(f)[0])[0]}_{i}.nii.gz")
        seg_crop.to_filename(
            rf"D:\verse\external_dataset_processed\crop\seg\{os.path.splitext(os.path.splitext(f)[0])[0]}_{i}.nii.gz")
        if mask_1_l != mask_0_l:
            print(os.path.splitext(f)[0])

        print(l_diff)
        print(mask_0_l)
        print(mask_1_l)
        print(mask_2_l)
        ratio = mask_1_l / mask_2_l
        print(ratio)
        result_list.append(ratio)
        mask1_list.append(mask_2_l)
print(result_list)
print(mask1_list)
