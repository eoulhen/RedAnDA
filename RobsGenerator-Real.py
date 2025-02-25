import matplotlib 
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import gsw
from datetime import datetime,date
import os as os
from matplotlib.gridspec import GridSpec
from numpy.fft import fft,fft2,fftfreq
from scipy.interpolate import interp2d
from sklearn.linear_model import LinearRegression
from scipy import fftpack
from tqdm import tqdm
from scipy.stats import linregress,norm
import dask as da
import xarray as xar
from sklearn.neighbors import BallTree
import glob as glob


timemin=1930
timemax=2002


# OPEN LEARNING PERIOD FIELD.

RC=xar.open_mfdataset('ISAS_TropicalPacific.nc')


#SPATIAL COORDINATES
X,Y = np.meshgrid(RC['longitude'].compute(),RC['latitude'].compute(),indexing='xy')
Nx = RC['longitude'].size
Ny = RC['latitude'].size
Nz = RC['depth'].size


# OPEN OBSERVATIONS

ENtemp2=xar.open_mfdataset('ENtemp_NoInterpNoFMoorAdjusted.nc').compute()

ENtemp2 = ENtemp2.where((ENtemp2['LONGITUDE']<X.max())&(ENtemp2['LONGITUDE']>X.min())&(ENtemp2['LATITUDE']<Y.max())&(ENtemp2['LATITUDE']>Y.min()),drop=True)
ENtemp2=ENtemp2.where(ENtemp2['POTM']>0,drop=True)


#TIME COORDINATES
JULREF = (np.datetime64('1950-01-01')-np.datetime64('0000-01-01')).astype('timedelta64[D]').astype('float64')
Btime=([np.datetime64(str(yr)+'-'+str(mth).zfill(2)+'-15') for yr in range(timemin,timemax+1) for mth in range(1,13) ]) 
Jtime=np.asanyarray((Btime-np.datetime64('1950-01-01')).astype('timedelta64[D]').astype('float64'))
Na=len(Btime)



X,Y = da.array.meshgrid(RC['longitude'],RC['latitude'],indexing='xy')
X=da.array.ravel(X)
Y=da.array.ravel(Y)
NxNy=X.size
NxNyNz=NxNy*Nz




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
Clim=RC['TEMP_polyfit_coefficients'].sel({'degree':0}).fillna(0).coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean().stack(NxNy=['latitude','longitude']).expand_dims({'Time':Na})+((Jtime)[:,np.newaxis,np.newaxis,np.newaxis]*RC['TEMP_polyfit_coefficients'].expand_dims({'Time':Na}).sel({'degree':1}).coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean()).stack(NxNy=['latitude','longitude']).rename('Clim')
Clim*=0
Xb = xar.DataArray(0*Clim.fillna(0).data+RC['T_Climato'].coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True).stack(NxNy=['latitude','longitude']).fillna(0).data[np.linspace(0,Na-1,Na,dtype='int')%12],coords=[np.arange(0,Na,1),RC['depth'],np.arange(0,NxNy,1)],dims=['Time','depth','NxNy'])

#Temporal interpolator
obstimeind=(ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['JULD'].dt.year-timemin)*12+(ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['JULD'].dt.month-1)
G=np.zeros((ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size,Na))
G[np.arange(0,ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size,1),obstimeind.data]=1-(ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['JULD'].dt.day)/31
G[np.arange(0,ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size,1),obstimeind.data+1]=(ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['JULD'].dt.day)/31

#Spatial interpolator
kdt = BallTree(da.array.concatenate((X[:,np.newaxis].compute_chunk_sizes(),Y[:,np.newaxis].compute_chunk_sizes()),axis=-1), leaf_size=50, metric='euclidean')
dist_knn, index_knn = kdt.query(da.array.concatenate((ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['LONGITUDE'].data[:,np.newaxis],ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['LATITUDE'].data[:,np.newaxis]),axis=-1), 4)
H = np.zeros((ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size,np.size(X[:])))
H[np.unravel_index((index_knn+np.arange(0,H.size,H.shape[1])[:,np.newaxis]).ravel(),H.shape)]=((1/dist_knn)/(1/dist_knn).sum(axis=1).reshape((-1,1))).ravel()
msk = RC['TEMP_detrend'].isel({'time':0,'depth':i}).stack({'NxNy':['latitude','longitude']}).isnull()
H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)]/=(1-H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)][:,msk].sum(1))[:,np.newaxis]
H[(msk.values)&(np.where(H!=0,True,False))] = 0
H[np.any(np.isnan(H),1)] = 0

#Anomalies
Tano = ENtemp2['POTM'].sel({'depth':i}).dropna(dim='N_PROF').values-(H.dot(Xb[:,i].T)*G).sum(1)


# First depth level, second step : evaluate the variance in 6°x6° grid cells
ENtemp3 = ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')

didivx=2
didivy=2

X,Y= da.array.meshgrid(RC.coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True)['longitude'],RC.coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True)['latitude'],indexing='xy')
X=da.array.ravel(X)#[~msk]
Y=da.array.ravel(Y)#[~msk]
Nx = np.unique(X.compute()).size
Ny = np.unique(Y.compute()).size
NxNy=Nx*Ny

#new spatial interpolator
kdt = BallTree(da.array.concatenate((X[:,np.newaxis].compute_chunk_sizes(),Y[:,np.newaxis].compute_chunk_sizes()),axis=-1), leaf_size=50, metric='euclidean')
dist_knn, index_knn = kdt.query(da.array.concatenate((ENtemp3.dropna(dim='N_PROF')['LONGITUDE'].data[:,np.newaxis],ENtemp3.dropna(dim='N_PROF')['LATITUDE'].data[:,np.newaxis]),axis=-1), 1)
H = np.zeros((ENtemp3.dropna(dim='N_PROF')['N_PROF'].size,np.size(X[:])))
H[np.unravel_index((index_knn+np.arange(0,H.size,H.shape[1])[:,np.newaxis]).ravel(),H.shape)]=((1/dist_knn)/(1/dist_knn).sum(axis=1).reshape((-1,1))).ravel()
msk = RC['TEMP_detrend'].isel({'time':0,'depth':i}).coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True).stack({'NxNy':['latitude','longitude']}).isnull()
H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)]/=(1-H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)][:,msk].sum(1))[:,np.newaxis]
H[(msk.values)&(np.where(H!=0,True,False))] = 0
H[np.any(np.isnan(H),1)] = 0

#print number of obs per position
print(np.unique(index_knn,return_counts=True)[1])
print(ENtemp3['N_PROF'].size/ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size)

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
    X=da.array.ravel(X)#[~msk]
    Y=da.array.ravel(Y)#[~msk]
    Nx = np.unique(X.compute()).size
    Ny = np.unique(Y.compute()).size
    NxNy=Nx*Ny

    Clim=RC['TEMP_polyfit_coefficients'].sel({'degree':0}).fillna(0).coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean().stack(NxNy=['latitude','longitude']).expand_dims({'Time':Na})+((Jtime)[:,np.newaxis,np.newaxis,np.newaxis]*RC['TEMP_polyfit_coefficients'].expand_dims({'Time':Na}).sel({'degree':1}).coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean()).stack(NxNy=['latitude','longitude']).rename('Clim')
    Clim*=0
    Xb = xar.DataArray(Clim.fillna(0).data+RC['T_Climato'].coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True).stack(NxNy=['latitude','longitude']).fillna(0).data[np.linspace(0,Na-1,Na,dtype='int')%12],coords=[np.arange(0,Na,1),RC['depth'],np.arange(0,NxNy,1)],dims=['Time','depth','NxNy'])


    obstimeind=(ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['JULD'].dt.year-timemin)*12+(ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['JULD'].dt.month-1)
    G=np.zeros((ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size,Na))
    G[np.arange(0,ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size,1),obstimeind.data]=1-(ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['JULD'].dt.day)/31
    G[np.arange(0,ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size,1),obstimeind.data+1]=(ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['JULD'].dt.day)/31

    from sklearn.neighbors import BallTree

    kdt = BallTree(da.array.concatenate((X[:,np.newaxis].compute_chunk_sizes(),Y[:,np.newaxis].compute_chunk_sizes()),axis=-1), leaf_size=50, metric='euclidean')


    dist_knn, index_knn = kdt.query(da.array.concatenate((ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['LONGITUDE'].data[:,np.newaxis],ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['LATITUDE'].data[:,np.newaxis]),axis=-1), 4)
    H = np.zeros((ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size,np.size(X[:])))
    H[np.unravel_index((index_knn+np.arange(0,H.size,H.shape[1])[:,np.newaxis]).ravel(),H.shape)]=((1/dist_knn)/(1/dist_knn).sum(axis=1).reshape((-1,1))).ravel()
    msk = RC['TEMP_detrend'].isel({'time':0,'depth':i}).stack({'NxNy':['latitude','longitude']}).isnull()
    H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)]/=(1-H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)][:,msk].sum(1))[:,np.newaxis]
    H[(msk.values)&(np.where(H!=0,True,False))] = 0
    H[np.any(np.isnan(H),1)] = 0

    Tano = ENtemp2['POTM'].sel({'depth':i}).dropna(dim='N_PROF').values-(H.dot(Xb[:,i].T)*G).sum(1)

    
    # ... Then the variance is evaluated in wider grid cells whose size vary depending the depth, as set by the following if condition
     
    ENtemp3 = ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')

    print(ENtemp3['N_PROF'].size/ENtemp2.sel({'depth':i}).dropna(dim='N_PROF')['N_PROF'].size)

    if i<=7:

        didivx=3
        didivy=2
        
    elif (i>7)&(i<=9) : 
    
        didivx=5
        didivy=4
        
    elif i>9 :     
        didivx=14*2
        didivy=9
    X,Y= da.array.meshgrid(RC.coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True)['longitude'],RC.coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True)['latitude'],indexing='xy')
    X=da.array.ravel(X)
    Y=da.array.ravel(Y)
    Nx = np.unique(X.compute()).size
    Ny = np.unique(Y.compute()).size
    NxNy=Nx*Ny



    kdt = BallTree(da.array.concatenate((X[:,np.newaxis].compute_chunk_sizes(),Y[:,np.newaxis].compute_chunk_sizes()),axis=-1), leaf_size=50, metric='euclidean')


    dist_knn, index_knn = kdt.query(da.array.concatenate((ENtemp3.dropna(dim='N_PROF')['LONGITUDE'].data[:,np.newaxis],ENtemp3.dropna(dim='N_PROF')['LATITUDE'].data[:,np.newaxis]),axis=-1), 1)
    H = np.zeros((ENtemp3.dropna(dim='N_PROF')
    H[np.unravel_index((index_knn+np.arange(0,H.size,H.shape[1])[:,np.newaxis]).ravel(),H.shape)]=((1/dist_knn)/(1/dist_knn).sum(axis=1).reshape((-1,1))).ravel()
    msk = RC['TEMP_detrend'].isel({'time':0,'depth':i}).coarsen({'latitude':didivy,'longitude':didivx},boundary="pad").mean(skipna=True).stack({'NxNy':['latitude','longitude']}).isnull()
    H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)]/=(1-H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)][:,msk].sum(1))[:,np.newaxis]
    H[(msk.values)&(np.where(H!=0,True,False))] = 0
    H[np.any(np.isnan(H),1)] = 0

    


    print(np.unique(index_knn,return_counts=True)[1])

    dof = np.zeros(NxNy)
    dof[np.unique(index_knn,return_counts=True)[0]] = np.unique(index_knn,return_counts=True)[1]

    Robs=xar.concat((Robs,(xar.DataArray(data=np.sqrt((Tano**2).dot(H)/(dof-1)).reshape((Ny,-1,1)),dims=['latitude','longitude','depth'],coords={'longitude':np.unique(X),'latitude':np.unique(Y),'depth':RC['depth'][i:i+1]})).interp_like(RC.isel({'depth':i}),method='nearest', kwargs={"fill_value": "extrapolate"}).bfill('latitude').bfill('longitude').where(RC['TEMP'][0].isel({'depth':i}).notnull(),np.nan)),dim='depth')

# Robs file is saved
Robs.rename('POTM_ERR').to_netcdf('Robs_TEMP.nc','w') 
