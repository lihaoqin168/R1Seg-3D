
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2


root = 'E:/02datasets/'
# npyfiles = ['M3D_Seg_npy/0011/s0091/image.npy',
#  'M3D_Seg_npy/0011/s1331/image.npy',
#  'M3D_Seg_npy/0011/s1293/image.npy',
#  'M3D_Seg_npy/0011/s1259/image.npy',
#  'M3D_Seg_npy/0011/s0394/image.npy',
#  'M3D_Seg_npy/0011/s0700/image.npy',
#  'M3D_Seg_npy/0011/s0612/image.npy',
#  'M3D_Seg_npy/0011/s0963/image.npy',
#  'M3D_Seg_npy/0011/s0156/image.npy',
#  'M3D_Seg_npy/0011/s1233/image.npy',
#  'M3D_Seg_npy/0009/3Dircadb1.14/image.npy',
#  'M3D_Seg_npy/0011/s0338/image.npy',
#  'M3D_Seg_npy/0013/word_0003/image.npy',
#  'M3D_Seg_npy/0010/FLARE22_Tr_0008_0000/image.npy',
#  'M3D_Seg_npy/0003/Case_00206_0000/image.npy',
#  'M3D_Seg_npy/0020/liver_79/image.npy',
#  'M3D_Seg_npy/0019/hepaticvessel_183/image.npy',
#  'M3D_Seg_npy/0011/s0712/image.npy',
#  'M3D_Seg_npy/0011/s1273/image.npy',
#  'M3D_Seg_npy/0011/s0312/image.npy',
#  'M3D_Cap_npy/ct_case/015757/Coronal_bone_window.npy',
#  'M3D_Cap_npy/ct_case/017817/Axial_liver_window.npy',
#  'M3D_Cap_npy/ct_case/007031/Coronal_C__portal_venous_phase.npy',
#  'M3D_Cap_npy/ct_case/003111/Axial_C__portal_venous_phase.npy',
#  'M3D_Cap_npy/ct_case/012354/Axial_bone_window.npy',
#  'M3D_Cap_npy/ct_case/010536/Coronal_C__portal_venous_phase.npy',
#  'M3D_Cap_npy/ct_case/000910/Axial_bone_window.npy',
#  'M3D_Cap_npy/ct_case/007101/Axial_C__portal_venous_phase.npy',
#  'M3D_Cap_npy/ct_case/009821/_3D_VR.npy',
#  'M3D_Cap_npy/ct_case/002919/Axial_lung_window.npy',
#  'M3D_Cap_npy/ct_case/008968/Sagittal_non_contrast.npy',
#  'M3D_Cap_npy/ct_case/002158/Sagittal_C__portal_venous_phase.npy']
npyfiles = ['M3D_Cap_npy/ct_case/007789/Axial_Zoomed___thin_cuts.npy','M3D_Cap_npy/ct_case/012804/Axial_non_contrast.npy']
for i in range(len(npyfiles)):
    image_np = np.load(root+npyfiles[i])
    image_pt = torch.from_numpy(image_np)
    if torch.isnan(image_pt).any():
        print('image_pt!!!!',npyfiles[i])
    #
    print('image_pt.shape[1]', image_pt.shape[1])
    slice = image_pt.shape[1]
    for j in range(slice):
        img = image_np[0][j]*256
        cv2.imwrite(str(i)+'-'+str(j)+'depthmap.png', img)
