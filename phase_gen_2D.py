import numpy as np
import pdb
# import plotly.plotly as py
import os
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.io as sp
from scipy import ndimage
import h5py
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interpn
from skimage import filters

def interpLS_2D(data,res_out):
    #2D interpolator
    res_in = data.shape

    x=np.linspace(0,1,res_in[0])
    y=np.linspace(0,1,res_in[1])

    XX = np.linspace(0,1,res_out[0])
    YY = np.linspace(0,1,res_out[1])

    coords_out = np.array(np.meshgrid(XX,YY))
    coords_out = np.transpose(coords_out,(2,1,0))

    data_out = interpn((x,y),data,coords_out)

    return data_out

def generateSH(rt,coil):
    # this program generates hte target field variation at the target field points using the cartesian forms for the spherical harmonic defined by the values in coil.
    # the first value in coil gives the degree of the spherical harmonic
    # the second value gives the degree of the spherical harmonic
    #
    # Usage:    B5 = target(rt,coil)
    #   rt - matrix containing the x,y,z coordinates of teh target field points
    #   Bt - target field values at the target field points
    #   coil - [order degree] of the target field
    #   Michael Poole, translated by LSacolick

    x = rt[:,0]
    y = rt[:,1]

    sh = np.empty_like(x)
    ii = 1
    nm = np.empty_like(coil)

    for n in range(0,coil+1):
        for m in range(0,n+1):
            numt = np.zeros_like(x)
            dent = 0
            for k in range(0,np.floor((n-m)/2).astype(int)+1):
#                 numt = numt + -1**k * (x+1j*y)**(k+m) * (x-1j*y)**k * z**(n-m-k-k)
                numt = numt + -1**k * (x+1j*y)**(k+m) * (x-1j*y)**k
                dent = dent + 2**(k+k+m) * np.math.factorial(k+m) * np.math.factorial(n-m-k-k)
    
            if m==0:
                sh = np.vstack((sh,np.math.factorial(n+m)*numt/dent))
                ii=ii+1
#                nm = np.vstack((nm,[n; m]))
            else:
                sh = np.vstack((sh,np.math.factorial(n+m)*np.real(numt/dent)))
                sh = np.vstack((sh,np.math.factorial(n+m)*np.imag(numt/dent)))
                ii=ii+2
#                nm = np.vstack((nm,[n; m]))
#                nm = np.vstack((nm,[n; -1*m]))                
                
                
    return sh

def gen_phase_2D(img):
    
    '''make a binary mask of the brain, then generate a articial phase map'''
    # make mask from the whole brain image - ZW
    val = filters.threshold_otsu(img)*0.8
    mask = img>=val
    
    mask = ndimage.binary_closing(mask)
    mask = ndimage.binary_opening(mask)

    #Set the # of training images to generate
    N_training = 1
    [Nx, Ny] = img.shape

    #Generate random sets of 3D spherical harmonic fields, give them random gaussian distribution of coefficients
    res = 100 #resolution of S.H. fields to be generated
    [X,Y] = np.meshgrid([np.arange(-1,1+2/(res-1),2/(res-1))],[np.arange(-1,1+2/(res-1),2/(res-1))])

    SHorder = 3

    rt = np.moveaxis(np.vstack((X.flatten(),Y.flatten())),1,0)
    
    SH = generateSH(rt,SHorder)
    SH = SH.reshape(SH.shape[0],res,res)

    #only keep 1st order, 2nd and some of 3rd order terms. could find out relevsnt ones from cedric
    indices = np.hstack((3,4,5,8,9,11,12))
    SH = SH[indices,:,:]
    SH = np.moveaxis(SH,0,-1)
    # print(SH.shape)

    #Generate random coefficients for the SH functions
    zero_order = np.pi*np.random.uniform(-1,1,(N_training,1)) #constant phase offset
    first_order = np.random.standard_normal((N_training,2)) #linears, scaling = 3 is arbitrary
    second_order = np.random.standard_normal((N_training,(SH.shape[2]-2))) #higher orders, scaling = 1 is arbitrary

    #apply random constanct phase
    imphase = np.ones_like(img)
    imphase = imphase*np.exp(1j*np.pi*zero_order)
    # print(imphase.shape)
    # first order (linears)       
    imphase = imphase * np.exp(1j * np.pi * first_order[0,0] * interpLS_2D(SH[:,:,0],[Nx, Ny]))
    imphase = imphase * np.exp(1j * np.pi * first_order[0,1] * interpLS_2D(SH[:,:,1],[Nx, Ny]))
    
    # second order 
    for iSH in range(2,SH.shape[2]):
        imphase = imphase * np.exp(1j * np.pi * second_order[0,iSH-3] * interpLS_2D(SH[:,:,iSH],[Nx, Ny])) 

    # noise anywhere outside the image will have uniformly distributed random phase.
    background = 1j * np.pi * np.random.uniform(low=-1.0, high=1.0, size=np.shape(imphase)) # background phase variation from -1 to 1 
    background =  np.exp(background)

     
    background[img==0] = 0
    imphase_com= mask*imphase+(1-mask)*background

    # apply phase to the original image
    img_comp = imphase_com * img
    
    return(img_comp.astype('complex64'))

if __name__ == "__main__":

    # read in the whole brain image - ZW
    filename = 'dl_research/projects/under_recon/M0_preDL_20171220_152834.nii'
    img = nib.load(filename)
    img = img.get_data()
    # print(img.shape)
    
    img_comp = gen_phase(img)

    ''' The ultimate input is the image, in 3D volume'''

