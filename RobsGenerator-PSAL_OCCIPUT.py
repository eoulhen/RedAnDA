import matplotlib 
matplotlib.rcParams.update({'font.size': 18})

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import gsw
from datetime import datetime,date
import os as os
from matplotlib.gridspec import GridSpec
from numpy.fft import fft,fft2,fftfreq
# importig movie py libraries
# from moviepy.editor import VideoClip
from scipy.interpolate import interp2d
# from moviepy.video.io.bindings import mplfig_to_npimage
from sklearn.linear_model import LinearRegression
from scipy import fftpack
from tqdm import tqdm
# analog data assimilation
from scipy.stats import linregress,norm
import dask as da
import xarray as xar
# from pydmd import DMD
import glob as glob





timemin=1961
timemax=1994


# OPEN LEARNING PERIOD FIELD. 

RC=xar.open_mfdataset('Pacific_OCCIPUT_Full2.nc').isel(depth=[0,1,2,3,4,5])#.isel(depth=[0,3,4,5,6,7,8]).isel(depth=[0,3,4,5,6,7,8])
RC=RC.where(RC['time2'].dt.year>=1995,drop=True)

RC['TEMP_polyfit_coefficients'] = RC['TEMP_polyfit_coefficients'].isel(time=0)
RC['T_Climato'] = RC['T_Climato'].isel(time=0)
RC['S_Climato'] = RC['S_Climato'].isel(time=0)
RC['PSAL_polyfit_coefficients'] = RC['PSAL_polyfit_coefficients'].isel(time=0)

RC = RC.merge(((RC['TEMP'].groupby('time2.month')-RC['T_Climato'])-RC['TEMP_polyfit_coefficients'].sel(degree=0)-RC['time']*RC['TEMP_polyfit_coefficients'].sel(degree=1)).rename('TEMP_detrend')).merge(((RC['PSAL'].groupby('time2.month')-RC['S_Climato'])-RC['PSAL_polyfit_coefficients'].sel(degree=0)-RC['time']*RC['PSAL_polyfit_coefficients'].sel(degree=1)).rename('PSAL_detrend'))

Clim1=RC['PSAL_polyfit_coefficients'].sel(degree=0)+RC['time']*RC['PSAL_polyfit_coefficients'].sel(degree=1)


# The trend is not considered here, to have it, remove the two next lines
RC['PSAL_detrend'].values=(RC['PSAL_detrend']+Clim1).values

RC['PSAL_polyfit_coefficients'].values = RC['PSAL_polyfit_coefficients'].values*0



# OPEN OBSERVATIONS 

ENPSAL2 = xar.open_mfdataset(glob.glob('ENsal_NoInterp*NoFMoor_OCCIPUT.nc')[0]).isel(depth=[0,1,2,3,4,5])

for i in range(0,3):
    
    ENPSAL2= xar.concat((ENPSAL2,xar.open_mfdataset(glob.glob('ENsal_NoInterp*NoFMoor_OCCIPUT.nc')[1:][i]).isel(depth=[0,1,2,3,4,5])),'N_PROF') 


X,Y = np.meshgrid(RC['longitude'].compute(),RC['latitude'].compute(),indexing='xy')
ENPSAL2 = ENPSAL2.where((ENPSAL2['LONGITUDE']<X.max())&(ENPSAL2['LONGITUDE']>X.min())&(ENPSAL2['LATITUDE']<Y.max())&(ENPSAL2['LATITUDE']>Y.min()),drop=True)



# Coordinates components

Nx = RC['longitude'].size
Ny = RC['latitude'].size
Nz = RC['depth'].size

JULREF = (np.datetime64('1950-01-01')-np.datetime64('0000-01-01')).astype('timedelta64[D]').astype('float64')

Btime=([np.datetime64(str(yr)+'-'+str(mth).zfill(2)+'-15') for yr in range(timemin,timemax+1) for mth in range(1,13) ]) 


Jtime=np.asanyarray((Btime-np.datetime64('1950-01-01')).astype('timedelta64[D]').astype('float64'))

Na=len(Btime)



X,Y = da.array.meshgrid(RC['longitude'],RC['latitude'],indexing='xy')
X=da.array.ravel(X)#[~msk]
Y=da.array.ravel(Y)#[~msk]
NxNy=X.size
NxNyNz=NxNy*Nz



Btime=([np.datetime64(str(yr)+'-'+str(mth).zfill(2)+'-15') for yr in range(timemin,timemax+1) for mth in range(1,13) ])
Btime=np.append(Btime,np.datetime64(str(timemax+1)+'-'+str(1).zfill(2)+'-15'))
Btime=np.append(Btime,np.datetime64(str(timemax+1)+'-'+str(2).zfill(2)+'-15'))
Na=len(Btime)
Jtime=np.asanyarray((Btime-np.datetime64('1950-01-01')).astype('timedelta64[D]').astype('float64'))


# First depth level, first step : create obs anomalies within 3°x3° grid cells

i=0

didivx=1
didivy=1

X,Y= da.array.meshgrid(RC.coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True)['longitude'],RC.coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True)['latitude'],indexing='xy')
X=da.array.ravel(X)
Y=da.array.ravel(Y)
Nx = np.unique(X.compute()).size
Ny = np.unique(Y.compute()).size
NxNy=Nx*Ny


# Generate climatology 
Clim=RC['PSAL_polyfit_coefficients'].sel({'degree':0}).fillna(0).coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean().stack(NxNy=['latitude','longitude']).expand_dims({'Time':Na})+((Jtime)[:,np.newaxis,np.newaxis,np.newaxis]*RC['PSAL_polyfit_coefficients'].expand_dims({'Time':Na}).sel({'degree':1}).coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean()).stack(NxNy=['latitude','longitude']).rename('Clim')
Xb = xar.DataArray(Clim.fillna(0).data+RC['S_Climato'].coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True).stack(NxNy=['latitude','longitude']).fillna(0).data[np.linspace(0,Na-1,Na,dtype='int')%12],coords=[np.arange(0,Na,1),RC['depth'],np.arange(0,NxNy,1)],dims=['Time','depth','NxNy'])

#Temporal interpolator
obstimeind=(ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['JULD'].dt.year-timemin)*12+(ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['JULD'].dt.month-1)
G=np.zeros((ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size,Na))
G[np.arange(0,ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size,1),obstimeind.data]=1-(ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['JULD'].dt.day)/31
G[np.arange(0,ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size,1),obstimeind.data+1]=(ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['JULD'].dt.day)/31

from sklearn.neighbors import BallTree

kdt = BallTree(da.array.concatenate((X[:,np.newaxis].compute_chunk_sizes(),Y[:,np.newaxis].compute_chunk_sizes()),axis=-1), leaf_size=50, metric='euclidean')


#Spatial interpolator
dist_knn, index_knn = kdt.query(da.array.concatenate((ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['LONGITUDE'].data[:,np.newaxis],ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['LATITUDE'].data[:,np.newaxis]),axis=-1), 4)
H = np.zeros((ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size,np.size(X[:])))
H[np.unravel_index((index_knn+np.arange(0,H.size,H.shape[1])[:,np.newaxis]).ravel(),H.shape)]=((1/dist_knn)/(1/dist_knn).sum(axis=1).reshape((-1,1))).ravel()
msk = RC['PSAL_detrend'].isel({'time':0,'depth':i}).stack({'NxNy':['latitude','longitude']}).isnull()
H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)]/=(1-H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)][:,msk].sum(1))[:,np.newaxis]
H[(msk.values)&(np.where(H!=0,True,False))] = 0
H[np.any(np.isnan(H),1)] = 0

#Anomalies
Tano = ENPSAL2['PSAL'].sel({'depth':i}).dropna(dim='N_PROF').values-(H.dot(Xb[:,i].T)*G).sum(1)



# First depth level, second step : evaluate the variance in 9°x6° grid cells

ENPSAL3 = ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')

didivx=3
didivy=2


X,Y= da.array.meshgrid(RC.coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True)['longitude'],RC.coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True)['latitude'],indexing='xy')
X=da.array.ravel(X)#[~msk]
Y=da.array.ravel(Y)#[~msk]
Nx = np.unique(X.compute()).size
Ny = np.unique(Y.compute()).size
NxNy=Nx*Ny

#new spatial interpolator
kdt = BallTree(da.array.concatenate((X[:,np.newaxis].compute_chunk_sizes(),Y[:,np.newaxis].compute_chunk_sizes()),axis=-1), leaf_size=50, metric='euclidean')
dist_knn, index_knn = kdt.query(da.array.concatenate((ENPSAL3.dropna(dim='N_PROF')['LONGITUDE'].data[:,np.newaxis],ENPSAL3.dropna(dim='N_PROF')['LATITUDE'].data[:,np.newaxis]),axis=-1), 1)
H = np.zeros((ENPSAL3.dropna(dim='N_PROF')['N_PROF'].size,np.size(X[:])))
H[np.unravel_index((index_knn+np.arange(0,H.size,H.shape[1])[:,np.newaxis]).ravel(),H.shape)]=((1/dist_knn)/(1/dist_knn).sum(axis=1).reshape((-1,1))).ravel()
msk = RC['PSAL_detrend'].isel({'time':0,'depth':i}).coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True).stack({'NxNy':['latitude','longitude']}).isnull()
H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)]/=(1-H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)][:,msk].sum(1))[:,np.newaxis]
H[(msk.values)&(np.where(H!=0,True,False))] = 0
H[np.any(np.isnan(H),1)] = 0


#print number of obs per position
print(np.unique(index_knn,return_counts=True)[1])
print(ENPSAL3['N_PROF'].size/ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size)


#degree of freedom
dof = np.zeros(NxNy)
dof[np.unique(index_knn,return_counts=True)[0]] = np.unique(index_knn,return_counts=True)[1]


#Error of representativity is computed for the first depth level
Robs=xar.DataArray(data=np.sqrt(((Tano)**2).dot(H)/(dof-1)).reshape((Ny,-1,1)),dims=['latitude','longitude','depth'],coords={'longitude':np.unique(X),'latitude':np.unique(Y),'depth':RC['depth'][i:i+1]}).interp_like(RC.isel({'depth':i}),method='nearest', kwargs={"fill_value": "extrapolate"}).bfill('latitude').bfill('longitude').where(RC['TEMP'][0].isel({'depth':i}).notnull(),np.nan)


#The loop repeats the process for all depth levels
for i in range(1,RC['depth'].size):
    
    
    #Anomalies are first evaluated...
    didivx=1
    didivy=1

    X,Y= da.array.meshgrid(RC.coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True)['longitude'],RC.coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True)['latitude'],indexing='xy')
    X=da.array.ravel(X)
    Y=da.array.ravel(Y)
    Nx = np.unique(X.compute()).size
    Ny = np.unique(Y.compute()).size
    NxNy=Nx*Ny

    Clim=RC['PSAL_polyfit_coefficients'].sel({'degree':0}).fillna(0).coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean().stack(NxNy=['latitude','longitude']).expand_dims({'Time':Na})+((Jtime)[:,np.newaxis,np.newaxis,np.newaxis]*RC['PSAL_polyfit_coefficients'].expand_dims({'Time':Na}).sel({'degree':1}).coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean()).stack(NxNy=['latitude','longitude']).rename('Clim')
    Xb = xar.DataArray(Clim.fillna(0).data+RC['S_Climato'].coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True).stack(NxNy=['latitude','longitude']).fillna(0).data[np.linspace(0,Na-1,Na,dtype='int')%12],coords=[np.arange(0,Na,1),RC['depth'],np.arange(0,NxNy,1)],dims=['Time','depth','NxNy'])


    obstimeind=(ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['JULD'].dt.year-timemin)*12+(ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['JULD'].dt.month-1)
    G=np.zeros((ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size,Na))
    G[np.arange(0,ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size,1),obstimeind.data]=1-(ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['JULD'].dt.day)/30
    G[np.arange(0,ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size,1),obstimeind.data+1]=(ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['JULD'].dt.day)/30

    from sklearn.neighbors import BallTree

    kdt = BallTree(da.array.concatenate((X[:,np.newaxis].compute_chunk_sizes(),Y[:,np.newaxis].compute_chunk_sizes()),axis=-1), leaf_size=50, metric='euclidean')


    dist_knn, index_knn = kdt.query(da.array.concatenate((ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['LONGITUDE'].data[:,np.newaxis],ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['LATITUDE'].data[:,np.newaxis]),axis=-1), 4)
    H = np.zeros((ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size,np.size(X[:])))
    H[np.unravel_index((index_knn+np.arange(0,H.size,H.shape[1])[:,np.newaxis]).ravel(),H.shape)]=((1/dist_knn)/(1/dist_knn).sum(axis=1).reshape((-1,1))).ravel()
    msk = RC['PSAL_detrend'].isel({'time':0,'depth':i}).stack({'NxNy':['latitude','longitude']}).isnull()
    H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)]/=(1-H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)][:,msk].sum(1))[:,np.newaxis]
    H[(msk.values)&(np.where(H!=0,True,False))] = 0
    H[np.any(np.isnan(H),1)] = 0

    Tano = ENPSAL2['PSAL'].sel({'depth':i}).dropna(dim='N_PROF').values-(H.dot(Xb[:,i].T)*G).sum(1)

    ENPSAL3 = ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')
    
    
    # ... Then the variance is evaluated in wider grid cells whose size vary depending the depth, as set by the following if condition
    
    print(ENPSAL3['N_PROF'].size/ENPSAL2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size)

    if i<4:

        didivx=3
        didivy=2
        
    else : 
    
        didivx=4
        didivy=3
    X,Y= da.array.meshgrid(RC.coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True)['longitude'],RC.coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True)['latitude'],indexing='xy')
    X=da.array.ravel(X)#[~msk]
    Y=da.array.ravel(Y)#[~msk]
    Nx = np.unique(X.compute()).size
    Ny = np.unique(Y.compute()).size
    NxNy=Nx*Ny



    kdt = BallTree(da.array.concatenate((X[:,np.newaxis].compute_chunk_sizes(),Y[:,np.newaxis].compute_chunk_sizes()),axis=-1), leaf_size=50, metric='euclidean')


    dist_knn, index_knn = kdt.query(da.array.concatenate((ENPSAL3.dropna(dim='N_PROF')['LONGITUDE'].data[:,np.newaxis],ENPSAL3.dropna(dim='N_PROF')['LATITUDE'].data[:,np.newaxis]),axis=-1), 1)
    H = np.zeros((ENPSAL3.dropna(dim='N_PROF')['N_PROF'].size,np.size(X[:])))
    H[np.unravel_index((index_knn+np.arange(0,H.size,H.shape[1])[:,np.newaxis]).ravel(),H.shape)]=((1/dist_knn)/(1/dist_knn).sum(axis=1).reshape((-1,1))).ravel()
    msk = RC['PSAL_detrend'].isel({'time':0,'depth':i}).coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True).stack({'NxNy':['latitude','longitude']}).isnull()
    H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)]/=(1-H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)][:,msk].sum(1))[:,np.newaxis]
    H[(msk.values)&(np.where(H!=0,True,False))] = 0
    H[np.any(np.isnan(H),1)] = 0

    print(np.unique(index_knn,return_counts=True)[1])



    dof = np.zeros(NxNy)
    dof[np.unique(index_knn,return_counts=True)[0]] = np.unique(index_knn,return_counts=True)[1]

    Robs=xar.concat((Robs,(xar.DataArray(data=np.sqrt((Tano**2).dot(H)/(dof-1)).reshape((Ny,-1,1)),dims=['latitude','longitude','depth'],coords={'longitude':np.unique(X),'latitude':np.unique(Y),'depth':RC['depth'][i:i+1]})).interp_like(RC.isel({'depth':i}),method='nearest', kwargs={"fill_value": "extrapolate"}).bfill('latitude').bfill('longitude').where(RC['TEMP'][0].isel({'depth':i}).notnull(),np.nan)),dim='depth')

    
# Robs file is saved    
Robs.rename('PSAL_ERR').to_netcdf('Robs_PSAL_OCCIPUT.nc','w')
