import numpy as np
import netCDF4 as nc
from datetime import datetime,date
import os as os
from numpy.fft import fft,fft2,fftfreq
from scipy.interpolate import interp2d
from sklearn.linear_model import LinearRegression
from scipy import fftpack
from tqdm import tqdm
from scipy.stats import linregress,norm
import dask as da
import xarray as xar
import glob as glob
from numpy.random import shuffle
from sklearn.neighbors import KDTree, BallTree
from RedAnDA_functions import AnDA_analog_forecasting, mk_stochastic, sample_discrete, resampleMultinomial, inv_using_SVDfrom numpy.linalg import pinv



#FUNCTION THAT CALCULATE HAVERSINE DISTANCE
def haversine_distances(X,Y):
    import dask.array as daa
    X=X*np.pi/180
    Y=Y*np.pi/180
    return 2*6371e3*daa.arcsin(daa.sqrt((daa.sin(.5*(Y[...,0][...,np.newaxis]-X[...,0][...,np.newaxis,:])))**2+daa.cos(X[...,0][...,np.newaxis,:])*daa.cos(Y[...,0][...,np.newaxis])*(daa.sin(.5*(Y[...,1][...,np.newaxis]-X[...,1][...,np.newaxis,:])))**2))

    

#RECONSTRUCTION PERIOD
timemin=1961
timemax=1994

#LOAD LEARNING PERIOD TO DETERMINE CLIMATOLOGY AND A PRIORI STATISTICS
RC=xar.load_dataset('Pacific_OCCIPUT_Full2.nc').isel(depth=[0,1,2,3,4,5])#.isel(depth=[0,3,4,5,6,7,8])
RC=RC.where(RC['time2'].dt.year>=1995,drop=True)
RC['TEMP_polyfit_coefficients'] = RC['TEMP_polyfit_coefficients'].isel(time=0)
RC['T_Climato'] = RC['T_Climato'].isel(time=0)
RC['S_Climato'] = RC['S_Climato'].isel(time=0)
RC['PSAL_polyfit_coefficients'] = RC['PSAL_polyfit_coefficients'].isel(time=0)
RC = RC.merge(((RC['TEMP'].groupby('time2.month')-RC['T_Climato'])-RC['TEMP_polyfit_coefficients'].sel(degree=0)-RC['time']*RC['TEMP_polyfit_coefficients'].sel(degree=1)).rename('TEMP_detrend')).merge(((RC['PSAL'].groupby('time2.month')-RC['S_Climato'])-RC['PSAL_polyfit_coefficients'].sel(degree=0)-RC['time']*RC['PSAL_polyfit_coefficients'].sel(degree=1)).rename('PSAL_detrend'))
Clim1=RC['TEMP_polyfit_coefficients'].sel(degree=0)+RC['time']*RC['TEMP_polyfit_coefficients'].sel(degree=1)
Clim2=np.zeros_like(Clim1) 
#NO TREND IN THE CLIMATOLOGY
RC['TEMP_detrend'].values=(RC['TEMP_detrend']+Clim1).values

RC['TEMP_polyfit_coefficients'].values = RC['TEMP_polyfit_coefficients'].values*0



#STANDARDIZED OBSERVATIONS DATASETS
ENtemp2 = xar.load_dataset(glob.glob('ENtemp_NoInterp*NoFMoor_OCCITPUT.nc')[0]).isel(depth=[0,1,2,3,4,5])
for i in range(0,3):
    ENtemp2= xar.concat((ENtemp2,xar.load_dataset(glob.glob('ENtemp_NoInterp*NoFMoor_OCCITPUT.nc')[1:][i]).isel(depth=[0,1,2,3,4,5])),'N_PROF')


ENtemp2=  ENtemp2.where((ENtemp2['LONGITUDE']<X.max())&(ENtemp2['LONGITUDE']>X.min())&(ENtemp2['LATITUDE']<Y.max())&(ENtemp2['LATITUDE']>Y.min()),drop=True)


#OBSERVATIONAL REPRESENTATIVITY ERROR
Robs=xar.load_dataarray('Robs_PSAL_OCCIPUT.nc').drop_vars(('time','time2'))
Robs= Robs.where(Robs!=np.inf,np.nan)
Robs2=xar.load_dataarray('Robs_TEMP_OCCIPUT.nc').drop_vars(('time','time2'))
Robs2= Robs2.where(Robs2!=np.inf,np.nan)
Robs = xar.Dataset({'POTM_ERR':Robs2,'PSAL_ERR':Robs})



#PARAMETERS OF THE SPATIAL GRID
Nx = RC['longitude'].size
Ny = RC['latitude'].size
Nz = RC['depth'].size

X,Y = np.meshgrid(RC['longitude'],RC['latitude'],indexing='xy')
NxNy=X.size
X=np.ravel(np.repeat(X[np.newaxis],Nz,0))#[~msk]
Y=np.ravel(np.repeat(Y[np.newaxis],Nz,0))#[~msk]
NxNyNz=NxNy*Nz




#NUMBER OF ITERATION FOR THE COST FUNCTION MINIZATION (SEE GOOD ET AL.2013)
iteration = 50 



#RADIUS OF THE TEMPORAL WINDOW FOR OBSERVATION ASSIMILATION
bch=.5



#CALCULATION OF THE A PRIORI STATISTICS
A1=(RC['TEMP_detrend'].fillna(0)).stack(NxNyNz=['depth','latitude','longitude']).values.T 
C=RC['TEMP'].fillna(0).std('time').stack(NxNyNz=['depth','latitude','longitude']).values 


#DEFINE CORRELATION RADII AND CORRELATION FUNCTION
Rref_1 = 300e3
Rref_z1 = 200
Rref_z2 = 100
obsinflfact=1.75
def C2(s,b):
    return (1+ b*s)*np.exp(-b*s)




#CALCULATION OF THE DISTANCE BETWEEN THE GRID CELLS
deglat = 111110;
dist = haversine_distances(np.array([Y,X]).T,np.array([Y,X]).T)
dist_z = np.repeat(np.repeat((RC['depth'].values[np.newaxis]-RC['depth'].values[:,np.newaxis]),NxNy,0),NxNy,1)

#A PRIORI COVARIANCE
B = (C[np.newaxis]*C[:,np.newaxis])*.5*(C2(dist,1/Rref_1)+C2(dist,1/Rref_2))*.5*(C2(dist_z,1/Rref_z1)+C2(dist_z,1/Rref_z2))


#DEFINE OPTIMAL INTERPOLATION FUNCTION. THE PARAMETERS ARE THE OBSERVATION (yo), THE LEARNING PERIOD (RC), THE ERROR OF REPRESENTATIVITY (ROBS), THE RECONSTRUCTION TIME VECTOR (Time), THE NAME OF THE SAVING FILE (savefile) AND IF SUCH FILE IS GENERATED (save), AND IF A LOADING OF A INTERUMPTED ANALYSIS IS NECESSARY (load)
def OI(yo, RC, Robs, Xb, Time , method='OI',savefile = 'oisavefile.npy',save=True,load =False):
  
    # dimensions
    T = len(Time)
    Nz = yo['depth'].size
    k0 = 0
    NxNyNz = NxNy*Nz

    # initialization
    if not load: 
        class x_hat:
            X_a = np.zeros([T,NxNyNz])
            P_a1 = np.zeros([T,NxNyNz])
            P_var1 = np.zeros([T,NxNyNz])
  
        time = Time

            
    else : 
        class x_hat:
            X_a = np.load(savefile,allow_pickle=True)[0]
            P_a1 = np.load(savefile,allow_pickle=True)[1]
            P_var1 = np.load(savefile,allow_pickle=True)[2]
        k0 = np.load(savefile,allow_pickle=True)[3][0]
        print(k0)

    if (method =='OI'):
        for k in tqdm(range(k0,T)):
            
            time = Time[k]

            supp_mth=bch
            
            mth_up=yo['JULD']<=(time+np.timedelta64(int(supp_mth*30),'D'))
            mth_dwn=yo['JULD']>=(time-np.timedelta64(int(supp_mth*30),'D'))
            _obs = yo.where(mth_up&mth_dwn,drop=True).stack({'ALL':['N_PROF','depth']}).dropna(dim='ALL')



            # NUMBER OF OBSERVATIONS FOR THE TIME STEP
            Nobs=_obs['ALL'].size
            
                       
                
            #NEAREST DEPTH INDICES AND VERTICAL INTERPOLATION MATRIX
            obstimedepth=_obs['depth']
            P=np.zeros((Nobs,Nz))
            P[np.arange(0,Nobs,1),obstimedepth]=1
            
        
            #NEAREST GRID CELLS INDICES AND INTERPOLATION MATRIX
            kdt = BallTree(da.array.concatenate((X[:NxNy,np.newaxis],Y[:NxNy,np.newaxis]),axis=-1), leaf_size=50, metric='euclidean')
            dist_knn, index_knn = kdt.query(da.array.concatenate((_obs['LONGITUDE'].data[:,np.newaxis],_obs['LATITUDE'].data[:,np.newaxis]),axis=-1), 4)
            H = np.zeros((Nobs,NxNy))#xar.DataArray(np.zeros((Nobs,NxNy)),coords=[np.arange(0,Nobs,1),np.arange(0,NxNy,1)],dims=['N_PROF','NxNy'])
            H[np.unravel_index((index_knn+np.arange(0,H.size,H.shape[1])[:,np.newaxis]).ravel(),H.shape)]=((1/dist_knn)/(1/dist_knn).sum(axis=1).reshape((-1,1))).ravel()
            H = np.repeat(H,repeats=Nz,axis=1)
            H = (H.reshape((Nobs,NxNy,Nz))*P[:,np.newaxis,:]).swapaxes(1,2).reshape((Nobs,NxNyNz))
            msk = Robs['POTM_ERR'].squeeze().stack({'NxNyNz':['depth','latitude','longitude']}).isnull()
            H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)]/=(1-H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)][:,msk].sum(1))[:,np.newaxis]
            H[(msk.values)&(np.where(H!=0,True,False))] = 0
            H[np.any(np.isnan(H),1)] = 0

            del dist_knn,

            
            #ERROR OF REPRESENTATIVITY IS DETERMINED IN OBSERVATION SPACE, WITH THE OPTION TO PERFORM THE CALCULATION WITH DASK, IN CASE THE COMPUTATION IS TOO HEAVY FOR THE MEMORY
            if (8*(_obs['ALL'].size**2))/((2**30) *1)<1:
                Ro=da.array.diag((H.dot((Robs['POTM_ERR'].fillna(0).stack(NxNyNz=['depth','latitude','longitude']))**2)))
            
            else: 
                Ro=da.array.diag((H.dot((Robs['POTM_ERR'].fillna(0).stack(NxNyNz=['depth','latitude','longitude']))**2)))
                Ro = Ro.rechunk(block_size_limit=5e8,balance=True)

                            
            #IDENTIFY OBSERVATIONS THAT FALL OUTSIDE THE TARGET GRID
            outofbound=H.dot((RC['TEMP_detrend'].isel(time=40)).fillna(0).stack(NxNyNz=['depth','latitude','longitude']))!=0

            #DISTANCE BETWEEN OBSERVATIONS
            xobs,yobs = _obs['LONGITUDE'].values[outofbound],_obs['LATITUDE'].values[outofbound]
            dist = haversine_distances(np.array([yobs,xobs]).T,np.array([yobs,xobs]).T)
            dist_z = np.abs(_obs['Depth'].values[outofbound][np.newaxis]-_obs['Depth'][outofbound].values[:,np.newaxis])
            

            #OBSERVATION-OBSERVATION COVARIANCE MATRIX
            Coo = H[outofbound].dot((C[np.newaxis]*C[:,np.newaxis]).dot(H[outofbound].T))*.5*(C2(dist,1/Rref_1)+C2(dist,1/Rref_2))*.5*(C2(dist_z,1/Rref_z1)+C2(dist_z,1/Rref_z2))

            
            #OBSERVATION-ANALYSIS DISTANCES AND COVARIANCE MATRIX
            dist_obs = haversine_distances(np.array([yobs,xobs]).T,np.array([Y,X]).T)
            dist_z2 = np.abs(np.repeat((RC['depth'].values[:,np.newaxis]-_obs['Depth'].values[outofbound][np.newaxis]),NxNy,0))
            Cao=  (C[np.newaxis]*C[:,np.newaxis]).dot(H[outofbound].T)*.5*(C2(dist_obs,1/Rref_1)+C2(dist_obs,1/Rref_2))*.5*(C2(dist_z2,1/Rref_z1)+C2(dist_z2,1/Rref_z2))
           
            
            Ro = Ro.compute()[outofbound][:,outofbound]
            H = H[outofbound]
            del P
            
                                
            #CALCULATION OF THE KALMAN GAIN
            W = Cao.dot(np.linalg.inv(Ro))
            Q = np.linalg.inv(H.dot(W)+np.identity(H.shape[0]))#
            K = Cao.dot(np.linalg.inv(Coo+Ro))
              
            y = _obs['POTM'].values[outofbound]
                 
                    
            
            #CALCULATION OF THE A PRIORI MEAN WITH PERSISTANCE (alpha parameter)
            alpha = .82            
            if k!=0:
                xf = np.ravel(Xb[k])+alpha*(x_hat.X_a[k-1]-np.ravel(Xb[k-1]))
            else :
                xf = Xb[k].stack({'NxNyNz':['depth','NxNy']}).values
                
                
            #FIRST GUESS
            x = xf.copy()
            
            #OBSERVATION ANOMALIES CALCULATION
            d= (y-(H.dot(xf.T)))
                    
                    

            #ITERATIVE UPDATE OF THE FIRST GUESS, MINIMIZING THE COST FUNCTION FOLLOWING ITS SLOPE
            for u in range(0,iteration):
                x+=K.dot(d)
                y-=Q.dot(d)
                d= y-(H.dot(x.T))
                
                            
            #FINAL VALUE OF THE ANALYSIS
            x_hat.X_a[k] = x
            
            # Analyse Error Matrix
            x_hat.P_a1[k] = np.diag(B - np.dot(K,Cao.T))

            # Percentage of variance
            # print(B.shape,np.dot(K,Cao.T).shape,x_hat.P_a1[k].shape,np.diag(C).shape)
            x_hat.P_var1[k] = np.sqrt(x_hat.P_a1[k]/np.diag(B)) # using average STD 

            
    return x_hat





#SET THE TIME COMPONENTS
JULREF = (np.datetime64('1950-01-01')-np.datetime64('0000-01-01')).astype('timedelta64[D]').astype('float64')
Btime=([np.datetime64(str(yr)+'-'+str(mth).zfill(2)+'-15') for yr in range(timemin,timemax+1) for mth in range(1,13) ]) 
Btime=np.append(Btime,np.datetime64(str(timemax+1)+'-'+str(1).zfill(2)+'-15'),)
Na = len(Btime)
Jtime=np.asanyarray((Btime-np.datetime64('1950-01-01')).astype('timedelta64[D]').astype('float64'))


#SET THE CLIMATOLOGY
Clim=(RC['TEMP_polyfit_coefficients'].sel({'degree':0}).expand_dims({'Time':Jtime.size})+(Jtime)[:,np.newaxis,np.newaxis,np.newaxis]*RC['TEMP_polyfit_coefficients'].sel({'degree':1}).expand_dims({'Time':Jtime.size})).fillna(0).stack(NxNy=['latitude','longitude']).rename('Clim').assign_coords({'Time':Btime})
Xb = (Clim.groupby('Time.month')+RC['T_Climato'].stack(NxNy=['latitude','longitude']).fillna(0)).reset_index('NxNy').compute()
max_tim=Na


#SET THE OBSERVATIONS INPUT
yo = ENtemp2





#START THE ANALYSIS
x_hat_oi = OI(yo, RC, Robs.rolling({'latitude':3,'longitude':3},center=True,min_periods=4).mean().where(Robs['POTM_ERR'].notnull()), Xb[:], Btime[:-1],savefile='oisavefile.npy',load=False )



#A XARRAY SET IS GENERATED WITH THE ANALYSIS RESULTS
OI=xar.Dataset({'values':(['time','NxNyNz'],x_hat_oi.X_a),'uncert':(['time','NxNyNz'],x_hat_oi.P_a1),'PCTVAR':(['time','NxNyNz'],x_hat_oi.P_var1)},coords = {'JULD':(['time'], Btime[:-1]),'depth':(['NxNyNz'],np.repeat(RC['depth'],NxNy).values),'lat':(['NxNyNz'],Y),'lon':(['NxNyNz'],X)}).set_index({"NxNyNz":['depth','lat','lon']}).unstack('NxNyNz')



OI['values'].attrs={'units': 'K', 'description': 'Analysed Absolute Potential Temperature Field'}
OI.uncert.attrs={'units': 'K^2', 'description': 'Uncertainty of the analysis (variance)'}
OI.PCTVAR.attrs={'units': '', 'description': 'Proportion of the a priori variance (multiply by 100 to obtain the percentage). If PCTVAR close to 1, it means no observations has modified the analysis.'}
OI.lat.attrs={'units': '°N'}
OI.lon.attrs={'units': '°E'}
OI.JULD.attrs={'description': 'Date',}
OI.depth.attrs={'description': 'Depth downward oriented', 'units':'meters'}
OI=OI.assign_attrs({"Description":"1961-1994 analysis, with Optimal Interpolation","How to use": " Remove the climatology found in RedAnDA files to obtain functions to obtain the full grid anomalies.","Temporal persistence":"0.82"})

#THE SET IS SAVED AS NETCDF FILE
OI.to_netcdf('OI_TEMP_OCCIPUT.nc','w') 