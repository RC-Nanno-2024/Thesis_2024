import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

path='path to the counts of Ca in .csv format'
data_Ca=pd.read_csv(path)
x=30
y=30
image_Ca=np.zeros((y,x), dtype='float')

for i in range(len(data_Ca)):
    x1=data_Ca.column[i]
    y1=data_Ca.row[i]
    image_Ca[y1,x1]=data_Ca.Ca_K[i]
image_Ca = image_Ca*600    
image_Ca = np.where((image_Ca>=4e+5), image_Ca, 0 )   

path='path to the counts of Mg in .csv format'
data_Mg=pd.read_csv(path)
x=30
y=30
image_Mg=np.zeros((y,x), dtype='float')

for i in range(len(data_Mg)):
    x1=data_Mg.column[i]
    y1=data_Mg.row[i]
    image_Mg[y1,x1]=data_Mg.Mg_K[i]

path='path to the counts of Sr in .csv format'
data_Sr=pd.read_csv(path)
x=30
y=30
image_Sr=np.zeros((y,x), dtype='float')

for i in range(len(data_Sr)):
    x1=data_Sr.column[i]
    y1=data_Sr.row[i]
    image_Sr[y1,x1]=data_Sr.Sr_L1[i]

path='path to the counts of Mn in .csv format'
data_Mn=pd.read_csv(path)
x=30
y=30
image_Mn=np.zeros((y,x), dtype='float')

for i in range(len(data_Mn)):
    x1=data_Mn.column[i]
    y1=data_Mn.row[i]
    image_Mn[y1,x1]=data_Mn.Mn_K[i]

path='path to the counts of Al in .csv format'
data_Al=pd.read_csv(path)
x=30
y=30
image_Al=np.zeros((y,x), dtype='float')

for i in range(len(data_Al)):
    x1=data_Al.column[i]
    y1=data_Al.row[i]
    image_Al[y1,x1]=data_Al.Al_K[i]  
    
path='path to the counts of Si in .csv format'
data_Si=pd.read_csv(path)
x=30
y=30
image_Si=np.zeros((y,x), dtype='float')

for i in range(len(data_Si)):
    x1=data_Si.column[i]
    y1=data_Si.row[i]
    image_Si[y1,x1]=data_Si.Si_K[i]  
    
image_Mg_Masked = ((image_Mg*image_Ca)/image_Ca)
image_Mg_Masked = np.where((image_Mg_Masked>25), image_Mg_Masked, 0.0001 )     

image_Sr_Masked = ((image_Sr*image_Ca)/image_Ca)
image_Sr_Masked = np.where((image_Sr_Masked>90), image_Sr_Masked, 0.00001 )     

image_Mn_Masked = ((image_Mn*image_Ca)/image_Ca)
image_Mn_Masked = np.where((image_Mn_Masked>2.11e+03), image_Mn_Masked, 0.00001 ) 

image_Al_Masked = ((image_Al*image_Ca)/image_Ca)
image_Al_Masked = np.where((image_Al_Masked>100), image_Al_Masked, 0.00001 ) 

image_Si_Masked = ((image_Si*image_Ca)/image_Ca)
image_Si_Masked = np.where((image_Si_Masked>1000), image_Si_Masked, 0.00001 ) 

plt.figure()
plt.imshow(image_Mg_Masked)
plt.colorbar(label="CPS_Mg", orientation="vertical")   

plt.figure()
plt.imshow(image_Sr_Masked)
plt.colorbar(label="CPS_Sr", orientation="vertical")   

plt.figure()
plt.imshow(image_Mn_Masked)
plt.colorbar(label="CPS_sr", orientation="vertical")   

plt.figure()
plt.imshow(image_Al_Masked)
plt.colorbar(label="CPS_Al", orientation="vertical") 

plt.figure()
plt.imshow(image_Si_Masked)
plt.colorbar(label="CPS_Si", orientation="vertical")

# image_3 = plt.imshow(image_3, cmap=plt.cm.RdBu)


i = Image.fromarray(image_Mg_Masked)
i.save('path to save processed Mg counts as image.tiff')
i = Image.fromarray(image_Sr_Masked)
i.save('path to save processed Sr counts as image.tiff')
i = Image.fromarray(image_Mn_Masked)
i.save('path to save processed Mn counts as image.tiff')
i = Image.fromarray(image_Al_Masked)
i.save('path to save processed Al counts as image.tiff')
i = Image.fromarray(image_Si_Masked)
i.save('path to save processed Si counts as image.tiff')
