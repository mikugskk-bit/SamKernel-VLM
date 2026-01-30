import pydicom as dicom
import os
import numpy as np
import matplotlib.pyplot as plt
import itk
import cv2
import skimage


CT_path = r'F:\EM0407\CT\CT'
CTList = os.listdir(CT_path)

CT_path_List = []
for l in CTList:
    p = os.path.join(CT_path,l)
    c1 = os.path.join(p,os.listdir(p)[0])
    CT_path_List.append(c1)

PET_path = r'F:\EM0407\PET'
PETList = os.listdir(PET_path)
PET_path_List = []
for i in PETList:
    fp = os.path.join(PET_path,i)
    file = os.path.join(fp,os.listdir(fp)[0])
    PET_path_List.append(file)

print(len(PET_path_List),len(CT_path_List))
# print(PET_path_List,CT_path_List)

total_len = 0
lens = []
cts = np.zeros((33459,160,160))
pets = np.zeros((33459,160,160))

for f in range(len(PET_path_List)):
    PathDicom = CT_path_List[f]
    LstFilesDCM = []

    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():
                LstFilesDCM.append(os.path.join(dirName,filename))


    PETpath = PET_path_List[f]
    pet = itk.imread(PETpath)
    pet_ar = itk.array_from_image(pet)

    print(PathDicom,PETpath)
    print(pet.GetOrigin(),pet.GetSpacing(),pet.GetDirection())

    ct = itk.imread(LstFilesDCM)
    ct_ar = itk.array_from_image(ct)
    print(ct.GetOrigin(),ct.GetSpacing(),ct.GetDirection())

    pet_tx,pet_ty,pet_th = pet.GetOrigin()
    ct_tx,ct_ty,ct_th = ct.GetOrigin()

    pet_dx,pet_dy,pet_dh = pet.GetSpacing()
    ct_dx,ct_dy,ct_dh = ct.GetSpacing()

    pet_ch = pet_th
    ct_ch = ct_th


    startIndex = 0
    endIndex = 0
    cur = 0
    stop = 0

    l1 = pet_th - ct_th
    print(l1,l1 / pet_dh)

    index_h_pet = []
    index_x_pet = []
    index_y_pet = []
    index_h_ct = []
    index_x_ct = []
    index_y_ct = []
    #xy index pet
    for i in range(pet_ar.shape[1]):
        index_x_pet.append(pet_tx+i*pet_dx)
    #xy index ct
    for i in range(ct_ar.shape[1]):
        index_x_ct.append(ct_tx+i*ct_dx)
    paired_xy_ct = []
    paired_xy_pet = []
    for i in range(len(index_x_pet)):
        for j in range(len(index_x_ct)):
            if np.abs(index_x_pet[i] - index_x_ct[j]) < 0.5 * ct_dx:
                paired_xy_ct.append(j)
                paired_xy_pet.append(i)
                break

    print(len(paired_xy_ct),len(paired_xy_pet))


    # h index pet
    for i in range(pet_ar.shape[0]):
        index_h_pet.append(pet_th-i*pet_dh)
    # h index ct
    for i in range(ct_ar.shape[0]):
        index_h_ct.append(ct_th-i*ct_dh)

    paired_h_ct = []
    findFirst = 0
    pet_start = 0
    pet_end = 0
    for i in range(len(index_h_pet)):
        find = 0
        for j in range(len(index_h_ct)):
            if np.abs(index_h_pet[i] - index_h_ct[j]) < 0.5 * pet_dh:
                paired_h_ct.append(j)
                find = 1
                break
        if find == 1 and findFirst == 0:
            findFirst = 1
            pet_start = i
        if findFirst == 1 and find == 0:
            pet_end = i
            break


    # print(paired_h_ct,pet_start,pet_end)

    pet_par = pet_ar[pet_start:pet_end,:,:]
    ct_par = ct_ar[paired_h_ct,:,:]
    ct_parx = ct_par[:,paired_xy_ct,:]
    ct_rpar = ct_parx[:,:,paired_xy_ct]
    pet_parx = pet_par[:,paired_xy_pet,:]
    pet_rpar = pet_parx[:,:,paired_xy_pet]
    # ct_rpar = np.zeros((ct_par.shape[0],192,192))
    # for k in range(ct_par.shape[0]):
    #     plt.subplot(1,3,1)
    #     plt.imshow(pet_rpar[k])

    #     # ct_rpar[k] = cv2.resize(ct_par[k].astype(float), (192, 192), interpolation=cv2.INTER_LINEAR)

    #     plt.subplot(1,3,2)
    #     plt.imshow(ct_rpar[k])

    #     plt.subplot(1,3,3)
    #     plt.imshow(ct_rpar[k]*10+pet_rpar[k])


    #     plt.show()
    if len(paired_xy_ct) != 160:
        ct_repar = np.zeros((ct_rpar.shape[0],160,160))
        pet_repar = np.zeros((ct_rpar.shape[0],160,160))
        for k in range(ct_rpar.shape[0]):
            ct_repar[k] = cv2.resize(ct_rpar[k].astype(float), (160, 160), interpolation=cv2.INTER_LINEAR)
            pet_repar[k] = cv2.resize(pet_rpar[k].astype(float), (160, 160), interpolation=cv2.INTER_LINEAR)
        pets[total_len:total_len+pet_rpar.shape[0]] = pet_repar
        cts[total_len:total_len+ct_rpar.shape[0]] = ct_repar
    else:
        pets[total_len:total_len+pet_rpar.shape[0]] = pet_rpar
        cts[total_len:total_len+ct_rpar.shape[0]] = ct_rpar

    print(pet_rpar.shape[0],ct_rpar.shape[0])

    total_len += pet_rpar.shape[0]

    lens.append(total_len)
    print(total_len)

np.savez('./ctpetraw.npz',CT = cts, PET = pets, lens = lens)

    # for i in range(pet_par.shape[0]):
    #     plt.subplot(1,2,1)
    #     plt.imshow(ct_par[i,:,:],cmap='gray')
    #     plt.subplot(1,2,2)
    #     plt.imshow(pet_par[i,:,:],cmap='gray')
    #     plt.show()

    # while stop == 0:
    #     if pet_ch > 0 and ct_ch > 0:
    #         stop = 1

    #     if 

    # plt.figure(dpi=100)
    # plt.axes().set_aspect('equal','datalim')
    # plt.set_cmap(plt.gray())
    # plt.pcolormesh(x,y,np.flipud(ArrayDicom[:,:,200]))
    # plt.show()