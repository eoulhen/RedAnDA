import numpy as np
import netCDF4 as nc
from datetime import datetime,date
import os as os
from numpy.fft import fft,fft2,fftfreq
# importig movie py libraries
from scipy.interpolate import interp2d
from sklearn.linear_model import LinearRegression
from scipy import fftpack
from tqdm import tqdm
# analog data assimilation
from scipy.stats import linregress,norm, multivariate_normal
import dask as da
import dask.array as daar
import xarray as xar
import glob as glob
from sklearn.neighbors import KDTree, BallTree
from numpy.linalg import pinv
from RedAnDA_functions import AnDA_analog_forecasting, mk_stochastic, sample_discrete, resampleMultinomial, inv_using_SVD


#PERIOD TO RECONSTRUCTION
timemin=1930
timemax=2001

#LOAD LEARNING PERIOD DATA
RC=xar.load_dataset('ISAS_TropicalPacific.nc').isel(depth=[0,1,2,3,4,5,6,7])
Clim1=RC['TEMP_polyfit_coefficients'].sel(degree=0)+RC['time']*RC['TEMP_polyfit_coefficients'].sel(degree=1)
Clim2=np.zeros_like(Clim1) 
RC['TEMP_polyfit_coefficients'].values = RC['TEMP_polyfit_coefficients'].values*0

Clim1=RC['PSAL_polyfit_coefficients'].sel(degree=0)+RC['time']*RC['PSAL_polyfit_coefficients'].sel(degree=1)
Clim2=np.zeros_like(Clim1)
RC['PSAL_polyfit_coefficients'].values = RC['PSAL_polyfit_coefficients'].values*0


#LOAD OBSERVATIONS
ENPSAL2 = xar.load_dataset(glob.glob('ENsal_NoInterp*0NoFMoorAdjusted.nc')[0]).isel(depth=[0,1,2,3,4,5,6,7])#.isel(depth=[0,1,2,3,5,7])
if 'POTM' in (list(xar.load_dataset(glob.glob('ENsal_NoInterp*0NoFMoorAdjusted.nc')[0]).isel(depth=[0,1,2,3,4,5,6,7]).keys())):
    ENPSAL2=ENPSAL2.rename({'POTM':'PSAL'})

for i in range(0,7):

    if 'POTM' in (list(xar.load_dataset(glob.glob('ENsal_NoInterp*0NoFMoorAdjusted.nc')[1:][i]).isel(depth=[0,1,2,3,4,5,6,7]).keys())):
        ENPSAL2= xar.concat((ENPSAL2,xar.load_dataset(glob.glob('ENsal_NoInterp*0NoFMoorAdjusted.nc')[1:][i]).isel(depth=[0,1,2,3,4,5,6,7]).rename({"POTM":"PSAL"})),'N_PROF')
    else : 
        ENPSAL2= xar.concat((ENPSAL2,xar.load_dataset(glob.glob('ENsal_NoInterp*0NoFMoorAdjusted.nc')[1:][i]).isel(depth=[0,1,2,3,4,5,6,7])),'N_PROF')



#LOAD OBSERVATIONS INSTRUMENTAL ERRORS
InstrErr2 = xar.load_dataset('ENsal_ERR.nc',).rename({"Err":"ErrS"})['ErrS']

EN= ENPSAL2.merge(InstrErr2).sortby('JULD')



EN = EN.where((EN['LONGITUDE']<RC['longitude'].max())&(EN['LONGITUDE']>RC['longitude'].min())&(EN['LATITUDE']<RC['latitude'].max())&(EN['LATITUDE']>RC['latitude'].min()),drop=True)

#SPATIAL COORDINATES VARIABLES
Nx = RC['longitude'].size
Ny = RC['latitude'].size
Nz = RC['depth'].size
X,Y = daar.meshgrid(RC['longitude'],RC['latitude'],indexing='xy')
X=daar.ravel(X)
Y=daar.ravel(Y)
NxNy=X.size
NxNyNz=NxNy*Nz


#OBSERVATIONAL ERROR OF REPRESENTATIVITY
Robs=xar.load_dataset('Robs_PSAL.nc').drop(('time','time2')).isel(depth=[0,1,2,3,4,5,6,7])
Robs= Robs.where(Robs['PSAL_ERR']!=np.inf,np.nan)






#NUMBER OF EOFS FOR THE ANALYSIS (L)
nEOF= 14

#NON RETAINED EOFS ASSOCIATED COVARIANCE HAVE NON-DIAGONAL CONTRIBUTION TO THE OBSERVATIONAL ERROR MATRI; IT DOES NOT CONCERN THE scd_trunct-th AND SUBSEQUENT MODES WHICH CONTRIBUTE ONLY IN TERMS OF VARIANCE IN THE DIAGONAL OF THE OSERVATION ERROR MATRIX. SEE Kaplan et al. (1997). TO REDUCE THIS PARAMETER SHOULD REDUCE THE COST OF THE COMPUTATION, AT THE EXPENSE OF SIMPLIFICATION OF THE ERROR COVARIANCE
scd_trunct=50


#INITIAL RADIUS OF THE TEMPORAL WINDOW WITHIN WHICH OBSERVATIONS ARE CONSIDERED (UNIT : MONTH)
bch=.5



#EOFS CALCULATION
Weights = np.sqrt(np.abs(np.cos(np.pi*RC['latitude']/180)))
A1=((RC['PSAL_detrend']).fillna(0)*Weights).stack(NxNyNz=['depth','latitude','longitude']).values.T 
C=A1.dot(A1.T)/(A1.shape[1]-1)
eigvalclim,eigclim=np.linalg.eigh(C)
eigvalclim,eigclim=eigvalclim[::-1][:],eigclim[:,::-1][:,:]

cutoff = (eigvalclim>eigvalclim.mean()).sum() 

#THIS FUNCTION REDISTRIBUTE TO VARIANCE OF THE FIRST EOFS OVER ALL OF THEM. SEE Kaplan et al. (1997) FOR MORE DETAILS.
def redsitri(x,beta):
     return (1-beta)*x+x.sum()*beta/x.size
eigvalclim2 = redsitri(eigvalclim[:cutoff],.01)
ww = np.sqrt(eigvalclim2)/np.sqrt(eigvalclim[:cutoff])
eigclim=eigclim[:,:cutoff]*ww
aclim = np.dot(A1.T,eigclim[:,:])

del C

#VARIANCE ASSOCIATED WITH NON RETAINED EOFS
Lamb,R=eigvalclim2[:nEOF],eigvalclim2[nEOF:]


#FUNCTION OF REDANDA, WITH INPUT BEING : 
#THE OBSERVATIONS (YO), THE ANALOG BACKCAST PARAMETERS (AF), THE LEARNING PERIOD (RC), THE ERROR OF REPRESENTATIVITY (ROBS), THE MONTHLY CLIMATOLOGY (XB), THE TIME COORDINATES OF THE RECONSTRUCTION PERIOD (TIME), THE SIZE OF THE ENSEMBLE (N), THE NAME OF THE SAVEFILE (SAVEFILE), THE DATA ASSIMILATION METHOD (METHOD), IF SAVING IS PERFORMED (SAVE), IF A LOADING IS NECESSARY BECAUSE OF AN INTERRUMPTED ANALYSIS (LOAD)
def AnDA_data_assimilation(yo, AF, RC, Robs, Xb, Time , N, savefile = 'savefile.npy',method ='AnEnKS',save=True,load =False):
    """ 
    Apply stochastic and sequential data assimilation technics using 
    model forecasting or analog forecasting. 
    """
    
    # dimensions
    n = AF.catalog.analogs.shape[1]
    T = len(Time)
    k0 = 0
    

    # initialization
    if not load: 
        class x_hat:
            part = np.zeros([T,N,n])
            weights = np.zeros([T,N])
            values = np.zeros([T,n])
#             loglik = np.zeros([T])
            mthcov = np.zeros([T])
        time = Time
        m_xa_part = np.zeros([T,N,n])
        xf_part = np.zeros([T,N,n])
        Pf = np.zeros([T,n,n])
        
            
    else : 
        class x_hat:
            part = np.load(savefile,allow_pickle=True)[0]
            weights = np.load(savefile,allow_pickle=True)[1]
            values = np.load(savefile,allow_pickle=True)[2]
            mthcov = np.load(savefile,allow_pickle=True)[3]

        Pf = np.load(savefile,allow_pickle=True)[4]
        xf_part = np.load(savefile,allow_pickle=True)[5]
        m_xa_part = np.load(savefile,allow_pickle=True)[6]
        k0 = np.load(savefile,allow_pickle=True)[7][0]
        

    # EnKF and EnKS methods
    if (method =='AnEnKF' or method =='AnEnKS'):
        for k in tqdm(range(k0,T)):
            
            time = Time[k]

            supp_mth=bch

            #EXTEND THE ASSIMILATION WINDOW UNTIL THERE IS ENOUGH OBSERVATIONS WITHIN (5000) OR THAT IT IS ONE YEAR LONG (bch = 6)
            mth_up=yo['JULD']<=(time+np.timedelta64(int(supp_mth*30),'D'))
            mth_dwn=yo['JULD']>=(time-np.timedelta64(int(supp_mth*30),'D'))
            _1990 = yo.where(mth_up&mth_dwn,drop=True).stack({'ALL':['N_PROF','depth']}).set_coords(('LONGITUDE','LATITUDE'))

            while (len(_1990['ALL'])<5000)&(supp_mth<=6) :

                supp_mth+=.5
                mth_up=yo['JULD']<=(time+np.timedelta64(int(supp_mth*30),'D'))
                mth_dwn=yo['JULD']>=(time-np.timedelta64(int(supp_mth*30),'D'))
                _1990 = yo.where(mth_up&mth_dwn,drop=True).stack({'ALL':['N_PROF','depth']}).set_coords(('LONGITUDE','LATITUDE'))

            Nobs = _1990['PSAL'].dropna(dim='ALL')['ALL'].size
            #THE SALINITY OBSERVATIONS OF THE PRESENT TIME STEP ARE SELECTED
            _1990 = (_1990['PSAL'].unstack({'ALL'})).stack({'ALL':['N_PROF','depth']}).dropna(dim='ALL')
            Nobs=_1990['ALL'].size
            #INSTRUMENTAL ERRORS ASSOCIATED WITH SELECTED OBSERVATIONS
            InstrErr2 = (_1990['ErrS']).where(_1990['PSAL'].notnull().values).dropna(dim='ALL')

            if _1990.N_PROF.size>0:
                #TEMPORAL INTERPOLATION MATRIX
                obstimeind=(-_1990['JULD'].dt.year+timemax)*12+11-(_1990['JULD'].dt.month-1)
                obstimeind=np.where(obstimeind==T,T-1,obstimeind)
                G=np.zeros((Nobs,T+1))
                G[np.arange(0,Nobs,1),obstimeind]=1

                #VERTICAL INTERPOLATION MATRIX
                obstimedepth=_1990['depth']
                P=np.zeros((Nobs,Nz))
                P[np.arange(0,Nobs,1),obstimedepth]=1


                #HORIZONTAL INTERPOLATION MATRIX
                kdt = BallTree(daar.concatenate((X[:,np.newaxis],Y[:,np.newaxis]),axis=-1), leaf_size=50, metric='euclidean')
                dist_knn, index_knn = kdt.query(daar.concatenate((_1990['LONGITUDE'].data[:,np.newaxis],_1990['LATITUDE'].data[:,np.newaxis]),axis=-1), 4)
                H = np.zeros((Nobs,NxNy))
                H[np.unravel_index((index_knn+np.arange(0,H.size,H.shape[1])[:,np.newaxis]).ravel(),H.shape)]=((1/dist_knn)/(1/dist_knn).sum(axis=1).reshape((-1,1))).ravel()
                H = np.repeat(H,repeats=Nz,axis=1)
                H = (H.reshape((Nobs,NxNy,Nz))*P[:,np.newaxis,:]).swapaxes(1,2).reshape((Nobs,NxNyNz))
                msk = RC.TEMP.isel(time=0).squeeze().stack({'NxNyNz':['depth','latitude','longitude']}).isnull()
                H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)]/=(1-H[np.any((msk.values)&(np.where(H!=0,True,False)),axis=1)][:,msk].sum(1))[:,np.newaxis]
                H[(msk.values)&(np.where(H!=0,True,False))] = 0
                H[np.any(np.isnan(H),1)] = 0
                Undo = np.diag(1/np.ravel((Weights).expand_dims({"depth":Nz,'longitude':Nx}).transpose('depth','latitude','longitude').values)) #UNWEIGHING MATRIX. 
                Undo[np.isinf(Undo)]=0
                del dist_knn,

                #CREATE INTERPOLATION MATRIX (OBS TO EOF SPACE)
                Eo=np.dot(H.dot(Undo),eigclim[:,:nEOF])


                #GENERATION OF THE OBSERVATION ERROR MATRIX
                Erro =((Robs['PSAL_ERR'])).fillna(0).stack(NxNyNz=['depth','latitude','longitude'])

                if (8*(_1990['ALL'].size**2))/((2**30) *1)<1: #IF A LOT OF OBS : USE DASK

                    preRo=H.dot(Undo).dot(eigclim[:,nEOF:scd_trunct].dot(np.diag(R[:scd_trunct-nEOF]).dot(eigclim[:,nEOF:scd_trunct].T.dot((H.dot(Undo)).T))))

                    postRo=H.dot(Undo).dot(eigclim[:,scd_trunct:].dot(np.diag(R[scd_trunct-nEOF:]).dot(eigclim[:,scd_trunct:].T.dot((H.dot(Undo)).T))))

                    Ro=daar.diag((H).dot(Erro**2))+preRo+np.eye(postRo.shape[0])*postRo
                  

                    del preRo,postRo

                else:
                    preRo=daar.dot(daar.dot(H.dot(Undo),eigclim[:,nEOF:scd_trunct].dot(np.diag(R[:scd_trunct-nEOF]).dot(eigclim[:,nEOF:scd_trunct].T))),(H.dot(Undo)).T).rechunk(block_size_limit=5e8,balance=True)

                    postRo=daar.dot(daar.dot(H.dot(Undo),eigclim[:,scd_trunct:].dot(np.diag(R[scd_trunct-nEOF:]).dot(eigclim[:,scd_trunct:].T))),(H.dot(Undo)).T).rechunk(preRo.chunks)

                    Ro=(daar.diag((H).dot(Erro**2))).rechunk(preRo.chunks)+preRo+daar.eye(postRo.shape[0]).rechunk(block_size_limit=5e8)*postRo

                    Ro = Ro.rechunk(block_size_limit=5e8,balance=True)
                    del preRo,postRo

                
                #GET RID OF OBSERVATIONS THAT FALL OUTSIDE THE GRID
                outofbound=H.dot(Erro)!=0

                #ADD INSTRUMENTAL ERRORS
                Ro+= np.diag(InstrErr2.values**2)

                #GENERATION OF OBSERVATION ANOMALIES    
                y= (_1990.values-(H.dot(Xb.stack({'NxNyNz':['depth','NxNy']}).values.T)*G).sum(1))[outofbound]
                


                Eo = Eo[outofbound]
                Ro = Ro.compute()[outofbound][:,outofbound]
                Nobs=Ro.shape[0]
                del H,G,P


            else:
                y = _1990.values
                print(y)

            #INITIALIZATION USING OBSERVATIONS ONLY           
            if k==0:
                

                invRo = np.linalg.inv(Ro)
                B = np.linalg.inv(Eo.T.dot(invRo).dot(Eo))
                xb = B.dot(Eo.T.dot(invRo).dot(y))
                xf = np.random.multivariate_normal(xb, B, N)
                    
                
            # update step (compute forecasts)   
            else:
                xf, m_xa_part_tmp = AnDA_analog_forecasting(x_hat.part[k-1,:,:], AF)
                m_xa_part[k,:,:] = m_xa_part_tmp  
                
            xf_part[k,:,:] = xf
            Ef = np.dot(xf.T,np.eye(N)-np.ones([N,N])/N)
            Pf[k,:,:] = np.dot(Ef,Ef.T)/(N-1)
            del Ef
            # analysis step (correct forecasts with observations)
            i_var_obs = np.where(~np.isnan(y))[0]       
            if (len(i_var_obs)>0)&(k>0):
                eps = np.random.normal(np.zeros(len(i_var_obs)),np.sqrt(np.diag(Ro))[i_var_obs],(N,Nobs))
                yf = np.dot(Eo[i_var_obs,:],xf.T).T
                
                
                SIGMA = np.dot(np.dot(Eo[i_var_obs,:],Pf[k,:,:]),Eo[i_var_obs,:].T)+Ro[np.ix_(i_var_obs,i_var_obs)]

                SIGMA_INV = np.linalg.inv(SIGMA)
                suppr = 0
                K = np.dot(np.dot(Pf[k,:,:],Eo[i_var_obs,:].T),SIGMA_INV)      
                d = y[i_var_obs][np.newaxis]+eps-yf
                x_hat.part[k,:,:] = xf + np.dot(d[:,:Nobs-suppr],K.T)           

                del SIGMA,SIGMA_INV,eps,yf,K,d,Ro,y,Eo,
            else:
                x_hat.part[k,:,:] = xf
            
            
                    
            x_hat.weights[k,:] = 1.0/N
            x_hat.values[k,:] = np.sum(x_hat.part[k,:,:]*x_hat.weights[k,:,np.newaxis],0)
       
                    
            #SAVE EVERY 10 STEPS                
            if save&(k%10==0):
                np.save(savefile,[x_hat.part,x_hat.weights,x_hat.values,x_hat.mthcov,Pf,xf_part,m_xa_part,np.array([k])],allow_pickle=True)
        

        # END AnEnKF AND FINAL SAVE       
        if save:
            np.save(savefile,[x_hat.part,x_hat.weights,x_hat.values,x_hat.mthcov,Pf,xf_part,m_xa_part,np.array([k])],allow_pickle=True)
        
        
        # EnKS method
        if (method == 'AnEnKS'):
            for k in tqdm(range(T-1,-1,-1)):           
                if k==T-1:
                    x_hat.part[k,:,:] = x_hat.part[T-1,:,:]
                else:
                    m_xa_part_tmp = m_xa_part[k+1,:,:]
                    tej, m_xa_tmp = AnDA_analog_forecasting(np.mean(x_hat.part[k,:,:],0)[np.newaxis], AF)
                    tmp_1 =(x_hat.part[k,:,:]-np.repeat(np.mean(x_hat.part[k,:,:],0)[np.newaxis],N,0)).T
                    tmp_2 = m_xa_part_tmp - m_xa_tmp
                    Ks = 1.0/(N-1)*np.dot(np.dot(tmp_1,tmp_2),inv_using_SVD(Pf[k+1,:,:],0.9999))                    
                    x_hat.part[k,:,:] = x_hat.part[k,:,:]+np.dot(x_hat.part[k+1,:,:]-xf_part[k+1,:,:],Ks.T)
                    
                x_hat.values[k,:] = np.sum(x_hat.part[k,:,:]*x_hat.weights[k,:,np.newaxis],0)
        # end AnEnKS  
    

    # error
    else :
        print("Error: choose DA.method between 'AnEnKF', 'AnEnKS', 'AnPF' ")
        quit()
    return x_hat



#TIME COORDINATES 
JULREF = (np.datetime64('1950-01-01')-np.datetime64('0000-01-01')).astype('timedelta64[D]').astype('float64')
Btime=([np.datetime64(str(yr)+'-'+str(mth).zfill(2)+'-15') for yr in range(timemin,timemax+1) for mth in range(1,13)
Btime=np.append(np.datetime64(str(timemin-1)+'-'+str(12).zfill(2)+'-15'),Btime)
Na = len(Btime)
Jtime=np.asanyarray((Btime-np.datetime64('1950-01-01')).astype('timedelta64[D]').astype('float64'))
        
        
#CLIMATOLOGY CALCULATION
Clim=(RC['PSAL_polyfit_coefficients'].sel({'degree':0}).expand_dims({'Time':Jtime.size})+(Jtime)[:,np.newaxis,np.newaxis,np.newaxis]*RC['PSAL_polyfit_coefficients'].sel({'degree':1}).expand_dims({'Time':Jtime.size})).fillna(0).stack(NxNy=['latitude','longitude']).rename('Clim').assign_coords({'Time':Btime})
Xb = ((Clim).groupby('Time.month')+RC['S_Climato'].stack(NxNy=['latitude','longitude']).fillna(0)).reset_index('NxNy').compute()


#OBSERVATION DATASET
yo = EN

        
#ANALOG FORECAST PARAMETERS
class catalog:
    analogs = aclim[:aclim.shape[0]][1:,:nEOF];
    successors = aclim[:aclim.shape[0]][:-1,:nEOF];

class AF:
    k = 170# number of analogs
    neighborhood = np.ones((nEOF,nEOF))
    catalog = catalog # catalog with analogs and successors
    regression = 'local_linear' # chosen regression ('locally_constant', 'increment', 'local_linear')
    sampling = 'gaussian' # chosen sampler ('gaussian', 'multinomial')

        
# run the analog data assimilation
x_hat_analog = AnDA_data_assimilation(yo, AF, RC, 1*Robs.rolling({'latitude':3,'longitude':3},center=True,min_periods=4).mean().where(Robs['PSAL_ERR'].notnull()), Xb[::-1], Btime[1:][::-1] , N=250, method ='AnEnKS',savefile='S_savefile.npy',load=True,)


#ERROR OF TRUNCATION ; VARIANCE ASSOCIATED WITH NON RETAINED EOFS
truncerr= (aclim[:,nEOF:].dot(eigclim[:,nEOF:].T)).std(0).reshape((Nz,Ny,Nx))/Weights.values[np.newaxis,:,np.newaxis]

#GENERATE THE XARRAY OBJECT CONTAINING THE RESULTS THEN SAVE IT AS A NETCDF FILE
redAnDA=xar.Dataset({'coeff':(['time','modes'],x_hat_analog.values[::-1]),'uncert':(['time','members','modes'],x_hat_analog.part[::-1]),'functions':(['depth','lat','lon','modes'],eigclim[:,:nEOF].reshape((Nz,Ny,Nx,nEOF))/Weights.values[np.newaxis,:,np.newaxis,np.newaxis]),'mthcov':(['time'],x_hat_analog.mthcov[::-1]),'truncerr':(['depth','lat','lon'],truncerr)},coords = {'JULD':(['time'], Btime[1:]),'Depth':(['depth'],RC['depth'].values),'Lat':(['lat'],RC['latitude'].data),'Lon':(['lon'],RC['longitude'].data)}).stack({"NxNy":['lat','lon']}).reset_index("NxNy", drop=False)
        
        
redanda=redanda.drop('lat').drop(('lon','mthcov')).assign({"Climato":(['month','depth','Lat','Lon'],RC.S_Climato.values)})

redanda.coeff.attrs={'units': '', 'description': 'Ensemble-mean of the temporal coefficients of the EOFs'}
redanda.uncert.attrs={'units': '', 'description': 'Ensemble of the temporal coefficients of the EOFs, used to compute analysis uncertainty'}
redanda.functions.attrs={'units': 'PSS', 'description': 'EOFs derived from the learning period salinity field'}
redanda.truncerr.attrs={'units': 'PSS', 'description': 'Truncation error : Standard deviation associated with non-retained EOFs'}
redanda.Climato.attrs={'units': 'PSS', 'description': 'Monthly climatology derived from the learning period'+str(RC.time2.dt.year.min().values)+'-'+str(RC.time2.dt.year.max().values)}
redanda.Lat.attrs={'units': '°N'}
redanda.Lon.attrs={'units': '°E'}
redanda.JULD.attrs={'description': 'Date',}
redanda.Depth.attrs={'description': 'Depth downward oriented', 'units':'meters'}
redanda=redanda.assign_attrs({"Description":str(timemin)+
"-"+str(timemax)+"  analysis, with UnivarS","How to use": " Apply the matrix product of the coeff with the functions to obtain the full grid anomalies.","Coefficient of inflation of the EOF spectrum" :"0.1","Minimal number of obs for analysis":"5000"})



redAnDA.to_netcdf('UnivarS_'+str(nEOF)+'modes.nc','w') 
