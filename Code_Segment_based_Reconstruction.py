import matplotlib.pyplot as plt
import numpy as np
import glob
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.tri as mtri
from scipy.spatial import ConvexHull
 
path="path to the images in .tiff format"
fns=glob.glob(path)
 
# Function that rebins the matrix from the tiff file
def rebin(arr, new_shape):
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
        new_shape[1], arr.shape[1] // new_shape[1])
        return arr.reshape(shape).mean(-1).mean(1)
 
# Initialize two matrices to save data
indices2=[]
values=[]
 
# Loop over all tiff files
for fn in fns:
                zind=1
                img=plt.imread(fn)  # Reads the tiff file
                imgrb=rebin(img,[736,736])  # Rebins the tiff image
                indices2.append(np.argwhere(imgrb!=0))   # Takes only the non-zero values of density
 
# indices2 is a list of arrays... here we copy it into a list called points
points=[]
z=0
for i in indices2:
                for j in range(len(i)):
                            points.append([i[j][0],i[j][1],z])
                z=z+1
#######################################
# Function to translate points to the center of mass
def move_to_center(points):
                # Calculate center of mass in x, y and z
                xmean=np.array(points).T[0].mean()
                ymean=np.array(points).T[1].mean()
                zmean=np.array(points).T[2].mean()
                points2=np.array(points)-[xmean,ymean,zmean]
                return points2
#########################################
# Functions to rotate vectors
def rotate_vector_z(vector, angle):
        x = vector[0] * math.cos(np.radians(angle)) - vector[1] * math.sin(np.radians(angle))
        y = vector[0] * math.sin(np.radians(angle)) + vector[1] * math.cos(np.radians(angle))
        z = vector[2]
        return [x, y, z]
 
def rotate_vector_y(vector, angle):
        x = vector[0] * math.cos(np.radians(angle)) - vector[2] * math.sin(np.radians(angle))
        y = vector[1]
        z = vector[0] * math.sin(np.radians(angle)) + vector[2] * math.cos(np.radians(angle))
        return [x, y, z]
 
def rotate_vector_x(vector, angle):
                x = vector[0]
                y = vector[1] * math.cos(np.radians(angle)) - vector[2] * math.sin(np.radians(angle))
                z = vector[1] * math.sin(np.radians(angle)) + vector[2] * math.cos(np.radians(angle))
                return [x, y, z]
##################################
# Functions to rotate whole lamella
def rotate_lamella_x(points,theta_step):
                rotated_points_x=[]
                for v in points:
                            rotated_points_x.append(rotate_vector_x(v,theta_step))
                return rotated_points_x
 
def rotate_lamella_y(points,theta_step):
                rotated_points_y=[]
                for v in points:
                            rotated_points_y.append(rotate_vector_y(v,theta_step))
                return rotated_points_y
 
def rotate_lamella_z(points,theta_step):
                rotated_points_z=[]
                for v in points:
                            rotated_points_z.append(rotate_vector_z(v,theta_step))
                return rotated_points_z
###############################
## Plot mesh of points
def plot_mesh(points):
                fig = plt.figure(1)
                ax = fig.add_subplot(projection='3d')
                # Extract x, y, and z coordinates from the data array
                x = np.array(points)[:, 0]
                y = np.array(points)[:, 1]
                z = np.array(points)[:, 2]
#####################################3
                # Create the scatter plot
                ax.scatter(x, y, z)
                plt.xlim(np.array(points).T[0].min()-10,np.array(points).T[0].max()+10)
                plt.ylim(np.array(points).T[1].min()-10,np.array(points).T[1].max()+10)
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                ax.set_zlim(np.array(points).T[2].min()-10,np.array(points).T[2].max()+10)
                plt.show()
 
def plot_vertices(points):
                hull=ConvexHull(points)
                fig = plt.figure(3)
                ax = fig.add_subplot(projection='3d')
                for i in hull.vertices:
                            ax.scatter(points[i][0],points[i][1],points[i][2])
                plt.xlim(np.array(points).T[0].min()-10,np.array(points).T[0].max()+10)
                plt.ylim(np.array(points).T[1].min()-10,np.array(points).T[1].max()+10)
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                ax.set_zlim(np.array(points).T[2].min()-10,np.array(points).T[2].max()+10)
                plt.show()
 
def plot_hull(points):
                hull=ConvexHull(points)
                # Plot the convex hull as lines
                points=np.array(points)
                fig = plt.figure(2)
                ax = fig.add_subplot(projection='3d')
                #ax.plot(points.T[0], points.T[1], points.T[2], "ko")
                for s in hull.simplices:
                            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
                            ax.plot3D(points[s, 0], points[s, 1], points[s, 2], c='grey')
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                ax.set_zlim(points.T[2].min()-10,points.T[2].max()+10)
                plt.show()
x=np.asarray(points).T[0]
y=np.asarray(points).T[1]
z=np.asarray(points).T[2]
 
# Determine the dimensions of the rectangular array based on the maximum x and y values
max_x = max(point[0] for point in points)
max_y = max(point[1] for point in points)
min_x = min(point[0] for point in points)
min_y = min(point[1] for point in points)
max_z = max(point[2] for point in points)
min_z = min(point[2] for point in points)
 
# Create the rectangular array and initialize with zeros
rectangular_array = np.zeros((int(max_x-min_x) + 1, int(max_y-min_y) + 1, int(max_z-min_z) + 1))
 
# Set the values at the specified coordinates to 255
for x, y, z in points:
                #print(x,y,z)
                rectangular_array[x-min_x, y-min_y, z-min_z] = 255
 
## Start of manipulations
lamella=move_to_center(points)
flat_y=rotate_lamella_y(lamella,45)
flat_x=rotate_lamella_x(flat_y,25)
flat_x=move_to_center(flat_x)
flat_z=rotate_lamella_z(flat_x, 60)
flat_final_1= move_to_center(flat_z)
 
def create_haxis_1(theta0,theta_step,R,zstep,rstep, Z):
   z=Z
   r=R
   vectors=[]
   num_vectors=np.array(np.arange(theta0, 90, theta_step))
   num=np.size(num_vectors)
   for i, th in enumerate(num_vectors):
               thetarad=np.radians(th)
               if i<=num-5:
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                r=r+rstep
                z=z+zstep
               else:
                r=r-rstep
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                z=z+zstep
               vectors.append([th, v])
               plt.scatter(th, i, c='orange')
               plt.scatter(r, i, c='blue')           
   return vectors
 
def create_haxis_2(theta0,theta_step,R,zstep,rstep, Z):              
   z=Z
   r=R
   vectors=[]
   num_vectors=np.array(np.arange(theta0, 120, theta_step))
   num=np.size(num_vectors)
   for i, th in enumerate(num_vectors):
               thetarad=np.radians(th)
               if i<=num-5:
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                r=r+rstep
                z=z+zstep
               else:
                r=r-rstep
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                z=z+zstep
               vectors.append([th, v])           
   return vectors
 
def create_haxis_3(theta0, theta_step,R,zstep,rstep, Z):
   z=Z
   r=R 
   vectors=[]
   num_vectors=np.array(np.arange(theta0, 150, theta_step))
   num=np.size(num_vectors)
   for i, th in enumerate(num_vectors):
               thetarad=np.radians(th)
               if i<=num-5:
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                r=r+rstep
                z=z+zstep
               else:
                r=r-rstep
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                z=z+zstep
               vectors.append([th, v])               
   return vectors
 
def create_haxis_4(theta0,theta_step,R,zstep,rstep, Z):
   z=Z
   r=R
   vectors=[]
   num_vectors=np.array(np.arange(theta0, 180, theta_step))
   num=np.size(num_vectors)
   for i, th in enumerate(num_vectors):
               thetarad=np.radians(th)
               if i<=num-5:
                v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                r=r+rstep
                z=z+zstep
               else:
                r=r-rstep
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                z=z+zstep
               vectors.append([th, v])              
   return vectors
 
def create_haxis_5(theta0,theta_step,R,zstep,rstep, Z):
   z=Z
   r=R
   vectors=[]
   num_vectors=np.array(np.arange(theta0, 210, theta_step))
   num=np.size(num_vectors)
   for i, th in enumerate(num_vectors):
               thetarad=np.radians(th)               
               if i<=num-5:
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                r=r+rstep
                z=z+zstep
               else:
                r=r-rstep
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                z=z+zstep
               vectors.append([th, v])
   return vectors
 
def create_haxis_6(theta0,theta_step,R,zstep,rstep, Z):
   z=Z
   r=R
   vectors=[]
   num_vectors=np.array(np.arange(theta0, 240, theta_step))
   num=np.size(num_vectors)
   for i, th in enumerate(num_vectors):
               thetarad=np.radians(th)
               if i<=num-5:
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                r=r+rstep
                z=z+zstep
               else:
                r=r-rstep
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                z=z+zstep
               vectors.append([th, v])
   return vectors
 
def create_haxis_7(theta0,theta_step,R,zstep,rstep, Z):
   z=Z
   r=R
   vectors=[]
   num_vectors=np.array(np.arange(theta0, 270, theta_step))
   num=np.size(num_vectors)
   for i, th in enumerate(num_vectors):
               thetarad=np.radians(th)
               if i<=num-5:
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                r=r+rstep
                z=z+zstep
               else:
                r=r-rstep
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                z=z+zstep
               vectors.append([th, v])
   return vectors
 
def create_haxis_8(theta0,theta_step,R,zstep,rstep, Z):
   z=Z
   r=R
   vectors=[]
   num_vectors=np.array(np.arange(theta0, 300, theta_step))
   num=np.size(num_vectors)
   for i, th in enumerate(num_vectors):
               thetarad=np.radians(th)
               if i<=num-5:
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                r=r+rstep
                z=z+zstep
               else:
                r=r-rstep
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                z=z+zstep
               vectors.append([th, v])
   return vectors
 
def create_haxis_9(theta0,theta_step,R,zstep,rstep, Z):
   z=Z
   r=R
   vectors=[]
   num_vectors=np.array(np.arange(theta0, 330, theta_step))
   num=np.size(num_vectors)
   for i, th in enumerate(num_vectors):
               thetarad=np.radians(th)
               if i<=num-5:
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                r=r+rstep
                z=z+zstep
               else:
                r=r-rstep
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                z=z+zstep
               vectors.append([th, v])
   return vectors
 
def create_haxis_10(theta0,theta_step,R,zstep,rstep, Z):
   z=Z
   r=R
   vectors=[]
   num_vectors=np.array(np.arange(theta0, 360, theta_step))
   num=np.size(num_vectors)
   for i, th in enumerate(num_vectors):
               thetarad=np.radians(th)
               if i<=num-5:
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                r=r+rstep
                z=z+zstep
               else:
                r=r-rstep
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                z=z+zstep
               vectors.append([th, v])
   return vectors  
 
def create_haxis_11(theta0,theta_step,R,zstep,rstep, Z):
   z=Z
   r=R
   vectors=[]
   num_vectors=np.array(np.arange(theta0, 390, theta_step))
   num=np.size(num_vectors)
   for i, th in enumerate(num_vectors):
               thetarad=np.radians(th)
               if i<=num-5:
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                r=r+rstep
                z=z+zstep
               else:
                r=r-rstep
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                z=z+zstep
               vectors.append([th, v])
   return vectors
def create_haxis_12(theta0,theta_step,R,zstep,rstep, Z):
   z=Z
   r=R
   vectors=[]
   num_vectors=np.array(np.arange(theta0, 420, theta_step))
   num=np.size(num_vectors)
   for i, th in enumerate(num_vectors):
               thetarad=np.radians(th)
               if i<=num-5:
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                r=r+rstep
                z=z+zstep
               else:
                r=r-rstep
            v=[math.cos(thetarad)*r,math.sin(thetarad)*r, z]
                z=z+zstep
               vectors.append([th, v])
   return vectors
################################################
### Input to create the Nannoconus
Z=0
vectors1=create_haxis_1(0,6,70,9,5,Z)  # Create first helicoidal axis
vectors2=create_haxis_2(30,6,70,9,5,Z)  # Create second helicoidal axis
vectors3=create_haxis_3(60,6,70,9,5,Z)  # Create second helicoidal axis
vectors4=create_haxis_4(90,6,70,9,5,Z)  # Create second helicoidal axis
vectors5=create_haxis_5(120,6,70,9,5,Z)  # Create first helicoidal axis
vectors6=create_haxis_6(150,6,70,9,5,Z)  # Create second helicoidal axis
vectors7=create_haxis_7(180,6,70,9,5,Z)  # Create second helicoidal axis
vectors8=create_haxis_8(210,6,70,9,5,Z)  # Create second helicoidal axis
vectors9=create_haxis_9(240,6,70,9,5,Z)  # Create second helicoidal axis
vectors10=create_haxis_10(270,6,70,9,5,Z)  # Create first helicoidal axis
vectors11=create_haxis_11(300,6,70,9,5,Z)  # Create second helicoidal axis
vectors12=create_haxis_12(330,6,70,9,5,Z)  # Create second helicoidal axis
 
nanno=[]
 
flat_h1=move_to_center(rotate_lamella_z(flat_final_1, 0))
flat_h2=move_to_center(rotate_lamella_z(flat_final_1, 0))
flat_h3=move_to_center(rotate_lamella_z(flat_final_1, 0))
flat_h4=move_to_center(rotate_lamella_z(flat_final_1, 0))
flat_h5=move_to_center(rotate_lamella_z(flat_final_1, 0))
flat_h6=move_to_center(rotate_lamella_z(flat_final_1, 0))
flat_h7=move_to_center(rotate_lamella_z(flat_final_1, 0))
flat_h8=move_to_center(rotate_lamella_z(flat_final_1, 0))
flat_h9=move_to_center(rotate_lamella_z(flat_final_1, 0))
flat_h10=move_to_center(rotate_lamella_z(flat_final_1, 0))
flat_h11=move_to_center(rotate_lamella_z(flat_final_1, 0))
flat_h12=move_to_center(rotate_lamella_z(flat_final_1, 0))
flat_h1_1=move_to_center(rotate_lamella_z(flat_final_1, 0))
flat_h1_2=move_to_center(rotate_lamella_z(flat_final_1, 0))
num=int(np.size(vectors1)/2)
 
for i, v in enumerate(vectors1):
        step_tilt=4.67
        delta_A=40
        delta_B=25
        if i%2==0:
         tilt=-35+i*step_tilt
         flat_2=rotate_lamella_x((rotate_lamella_y(flat_h1, tilt)), delta_A)
        else:
          tilt=-35+i*tilt-increment
          flat_2=rotate_lamella_x((rotate_lamella_y(flat_h1, tilt)), delta_B)
        flatr1=np.array(rotate_lamella_z(flat_2, v[0]))+v[1]
        nanno.append(flatr1)
        plt.scatter(tilt, i, c='red')
 
for i, v in enumerate(vectors2):
         tilt-increment=4.67
         delta_A=40
         delta_B=25
         if i%2==0:          
          tilt=-35+i*tilt-increment
          flat_2=rotate_lamella_x((rotate_lamella_y(flat_h2, tilt)), delta_B)
          plt.scatter(delta_A, i, c="yellow")
         else:
          tilt=-35+i*tilt-increment
          flat_2=rotate_lamella_x((rotate_lamella_y(flat_h2, tilt)), delta_A)
          plt.scatter(delta_B, i, c="green")
         flatr1=np.array(rotate_lamella_z(flat_2, v[0]))+v[1]
         nanno.append(flatr1)
 
for i, v in enumerate(vectors3):
        tilt-increment=4.67
        delta_A=40
        delta_B=25
        if i%2!=0:
         tilt=-35+i*tilt-increment
         flat_2=rotate_lamella_x((rotate_lamella_y(flat_h3, tilt)), delta_A)
        else:
          tilt=-35+i*tilt-increment
          flat_2=rotate_lamella_x((rotate_lamella_y(flat_h3, tilt)), delta_B)
        flatr1=np.array(rotate_lamella_z(flat_2, v[0]))+v[1]
        nanno.append(flatr1)
 
for i, v in enumerate(vectors4):
        tilt-increment=4.67
        delta_A=40
        delta_B=25
        if i%2==0:
         tilt=-35+i*tilt-increment
         flat_2=rotate_lamella_x((rotate_lamella_y(flat_h4, tilt)), delta_B)
        else:
          tilt=-35+i*tilt-increment
          flat_2=rotate_lamella_x((rotate_lamella_y(flat_h4, tilt)), delta_A)
        flatr1=np.array(rotate_lamella_z(flat_2, v[0]))+v[1]
        nanno.append(flatr1)
 
for i, v in enumerate(vectors5):
        tilt-increment=4.67
        delta_A=40
        delta_B=25
        if i%2!=0:
         tilt=-35+i*tilt-increment
         flat_2=rotate_lamella_x((rotate_lamella_y(flat_h5, tilt)), delta_A)
        else:
          tilt=-35+i*tilt-increment
          flat_2=rotate_lamella_x((rotate_lamella_y(flat_h5, tilt)), delta_B)
        flatr1=np.array(rotate_lamella_z(flat_2, v[0]))+v[1]
        nanno.append(flatr1)
 
for i, v in enumerate(vectors6):
        tilt-increment=4.67
        delta_A=40
        delta_B=25
        if i%2==0:
         tilt=-35+i*tilt-increment
         flat_2=rotate_lamella_x((rotate_lamella_y(flat_h6, tilt)), delta_B)
        else:
          tilt=-35+i*tilt-increment
          flat_2=rotate_lamella_x((rotate_lamella_y(flat_h6, tilt)), delta_A)
        flatr1=np.array(rotate_lamella_z(flat_2, v[0]))+v[1]
        nanno.append(flatr1)
 
for i, v in enumerate(vectors7):
        tilt-increment=4.67
        delta_A=40
        delta_B=25
        if i%2!=0:
         tilt=-35+i*tilt-increment
         flat_2=rotate_lamella_x((rotate_lamella_y(flat_h7, tilt)), delta_A)
        else:
          tilt=-35+i*tilt-increment
          flat_2=rotate_lamella_x((rotate_lamella_y(flat_h7, tilt)), delta_B)
        flatr1=np.array(rotate_lamella_z(flat_2, v[0]))+v[1]
        nanno.append(flatr1)
 
for i, v in enumerate(vectors8):
        tilt-increment=4.67
        delta_A=40
        delta_B=25
        if i%2==0:
         tilt=-35+i*tilt-increment
         flat_2=rotate_lamella_x((rotate_lamella_y(flat_h8, tilt)), delta_B)
        else:
          tilt=-35+i*tilt-increment
          flat_2=rotate_lamella_x((rotate_lamella_y(flat_h8, tilt)), delta_A)
        flatr1=np.array(rotate_lamella_z(flat_2, v[0]))+v[1]
        nanno.append(flatr1)
 
for i, v in enumerate(vectors9):
        tilt-increment=4.67
        delta_A=40
        delta_B=25
        if i%2!=0:
         tilt=-35+i*tilt-increment
         flat_2=rotate_lamella_x((rotate_lamella_y(flat_h9, tilt)), delta_A)
        else:
          tilt=-35+i*tilt-increment
          flat_2=rotate_lamella_x((rotate_lamella_y(flat_h9, tilt)), delta_B)
        flatr1=np.array(rotate_lamella_z(flat_2, v[0]))+v[1]
        nanno.append(flatr1)
 
for i, v in enumerate(vectors10):
        tilt-increment=4.67
        delta_A=40
        delta_B=25
        if i%2==0:
         tilt=-35+i*tilt-increment
         flat_2=rotate_lamella_x((rotate_lamella_y(flat_h10, tilt)), delta_B)
        else:
          tilt=-35+i*tilt-increment
          flat_2=rotate_lamella_x((rotate_lamella_y(flat_h10, tilt)), delta_A)
   flatr1=np.array(rotate_lamella_z(flat_2, v[0]))+v[1]
        nanno.append(flatr1)
 
for i, v in enumerate(vectors11):
   tilt-increment=4.67
   delta_A=40
   delta_B=25
        if i%2!=0:
         tilt=-35+i*tilt-increment
         flat_2=rotate_lamella_x((rotate_lamella_y(flat_h11, tilt)), delta_A)
        else:
          tilt=-35+i*tilt-increment
          flat_2=rotate_lamella_x((rotate_lamella_y(flat_h11, tilt)), delta_B)
        flatr1=np.array(rotate_lamella_z(flat_2, v[0]))+v[1]
        nanno.append(flatr1)
 
for i, v in enumerate(vectors12):
        tilt-increment=4.67
        delta_A=40
        delta_B=25
        if i%2==0:
         tilt=-35+i*tilt-increment
         flat_2=rotate_lamella_x((rotate_lamella_y(flat_h12, tilt)), delta_B)
        else:
          tilt=-35+i*tilt-increment
          flat_2=rotate_lamella_x((rotate_lamella_y(flat_h12, tilt)), delta_A)
        flatr1=np.array(rotate_lamella_z(flat_2, v[0]))+v[1]
        nanno.append(flatr1)
########################################################################
# Create the rectangular array and initialize with zeros
# Determine the dimensions of the rectangular array based on the maximum x and y values
nanno=np.reshape(np.array(nanno),(int((np.size(nanno)/3)),3))
max_x = round(nanno.T[0].max())
min_x = round(nanno.T[0].min())
max_y = round(nanno.T[1].max())
min_y = round(nanno.T[1].min())
max_z = round(nanno.T[2].max())
min_z = round(nanno.T[2].min())
rectangular_array = np.zeros((int(max_x-min_x) + 1, int(max_y-min_y) + 1, int(max_z-min_z) + 1))
# Set the values at the specified coordinates to 255
for x, y, z in nanno:
                rectangular_array[round(x-min_x), round(y-min_y), round(z-min_z)] = 255
np.save('E:\\Single lamella\\Analysis_Globulus\\New_Analysis\\nanno.npy',rectangular_array)
array = np.load('E:\\Single lamella\\Analysis_Globulus\\New_Analysis\\nanno.npy')
array = array.astype(np.float32)
from ORSModel import createChannelFromNumpyArray
array= createChannelFromNumpyArray(array)
array.publish()
 
Summary
The steps that followed in this code in the process of the skeletal reconstruction of Nannoconus globulus are given below with the associated line numbers:
Line numbers
	Working steps
	(1-35)
	Loading the images of the segmented lamella in an 3D-array of numbers
	(38-63)
	Function created to for translation and rotation of the lamellae
	(46-82)
	Functions created to rotate vectors
	(84-151)
	A mesh and hull are created from the 3D-array to reduce its size
	(154-159)
	The lamellae (hull) is alienated parallel to the basal plane, with its center at the origin of the co-ordinate system and
	(161-403)
	12 series of points (r, ɵ, z) are created with the values of different parameters mentioned in tables-1 of section-chapter
	(407-580)
	The full skeleton is created by assigning the lamella with rotation creating tilt and inclinations. This saved as a 3D array
	(582-602)
	The 3D array of the reconstructed skeleton is visualized as 3D object in the software (Dragonfly)