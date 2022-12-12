import os.path

import SimpleITK as sitk
import numpy as np
import glob
import matplotlib.pyplot as plt


def img_plot(img):
    plt.figure()  # 设置窗口大小
    plt.suptitle('Multi_Image')  # 图片名称
    for i in range(128):
        plt.subplot(12, 12, i + 1)
        plt.imshow(img[:, :, i])
    plt.show()


def offsetVolume(vol, inter):
    inputsize = vol.GetSize()
    i_list = [
        [0, 0, 0],
        [inputsize[0], 0, 0],
        [0, inputsize[1], 0],
        [inputsize[0], inputsize[1], 0],

        [0, 0, inputsize[2]],
        [0, inputsize[1], inputsize[2]],
        [inputsize[0], 0, inputsize[2]],
        [inputsize[0], inputsize[1], inputsize[2]]
    ]

    p_list = [vol.TransformIndexToPhysicalPoint(i) for i in i_list]
    origin = list(p_list[0])
    corner = list(p_list[0])
    for p in p_list:
        for i in [0, 1, 2]:
            if p[i] > corner[i]:
                corner[i] = p[i]
            if p[i] < origin[i]:
                origin[i] = p[i]

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(vol)

    resampler.SetOutputOrigin(origin)
    resampler.SetSize([int((corner[p] - origin[p]) / vol.GetSpacing()[p]) for p in [0, 1, 2]])

    resampler.SetOutputDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    if str.lower(inter) == 'nearest':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif str.lower(inter) == 'linear':
        resampler.SetInterpolator(sitk.sitkLinear)
    newvol = resampler.Execute(vol)

    return newvol


file_list = glob.glob(r"D:\verse\external_dataset_processed\raw\img\*.nii.gz")
target = r"D:\verse\external_dataset_processed\align2\img"

for i in file_list:
    print(f"===={i}====")

    vol = sitk.Image(sitk.ReadImage(i))

    fn = os.path.split(i)[1]
    print("Pre Size:", vol.GetSize())
    print("Pre Origin:", vol.GetOrigin())
    print("Pre Direction:", vol.GetDirection())

    # linear for CT image, nearest for segmentation mask
    newvol = offsetVolume(vol, inter='linear')

    print("Post Size:", newvol.GetSize())
    print("Post Origin:", newvol.GetOrigin())
    print("Post Direction:", newvol.GetDirection())

    sitk.WriteImage(newvol, os.path.join(target, fn))
