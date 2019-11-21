import os, glob
import numpy as np
import pandas as pd
from scipy.stats import uniform,norm
import warnings; warnings.simplefilter("ignore")

from myFun import SoilRoot, InitialState, f_PM_hydraulics,dailyAvg
from myFun import Constraint_p50, Constraint_b0,AMIS_proposal_constraint,AMIS_prop_loglik,ReplaceItems,SwapChains

#======= for running on server ==========
#arrayid = int(os.environ['SLURM_ARRAY_TASK_ID']) # 0-119
#niter = 2000
#numchunck = 10
#========================================
    
#===== for running on local machine ======
arrayid = 0
niter = 1
numchunck = 2
#=========================================

sitename = 'US-Me2'
# Site specific parameters
soil_texture = 2; root_type = 4
root_depth = 2; canopy_height = 33; tower_height = 47 # in meters
nobsinaday = 48 # number of observations per day

df = pd.read_csv(sitename+'.csv')[:nobsinaday*365] # use the first year to retrieve parameters, as an example
SRparas = SoilRoot(soil_texture,root_type,root_depth,canopy_height,tower_height,24*3600/nobsinaday,1,0)
Init = InitialState(-0.05,-0.1,-0.2,-0.1,df['SOILM'][0],df['SOILM2'][0])

#%  Forward Run

#%% AMIS to find parameters
def Gaussian_loglik(theta):
    theta_complete = theta*scale 
    E,PSIL = f_PM_hydraulics(df,SRparas,theta_complete)
    yhat = dailyAvg(E,nobsinaday)[~discard]*1e3
    return np.sum(norm.logpdf(observed_day_valid,yhat,theta[varid]*scale[varid]))


chains_per_site = 10
outpath = 'Output/'

fid = int(arrayid/chains_per_site)
chainid = arrayid-fid*chains_per_site

mVPD = np.mean(df['VPD'][(df['P']==0) & (df['RNET']>0)])
discard = dailyAvg(df['P'],nobsinaday)>10/nobsinaday # rainy days
discard[:30*2] = True # warm up period
observed_day_valid = dailyAvg(df['ET'],nobsinaday)[~discard]

varnames = ['gpmax','p50','aa','lww','b0','sigma','calib','bc']
varid = varnames.index('sigma')
varnames.append('loglik')
p = int(len(varnames)-1)

lowbound = np.array([1,-10,1,500,-2,0.01,0.5,SRparas.sw2])
upbound = np.array([10000,-0.5,8,10000,0,2,1.5,SRparas.ssat2])
scale = np.max(abs(np.column_stack([lowbound,upbound])),axis=1)

# Initialize, sample in a rescaled world; 
# Scale is only needed when evaluating likelihood and exporting results
lowbound = lowbound/scale
upbound = upbound/scale
bounds = (lowbound, upbound, scale)

# Parameters of AMIS, see Ji and Schmidler 
# Temperatures used for tempering
temps = 2**np.arange(0,6,2)                                                         
mu = [np.mean(np.column_stack([lowbound,upbound]),axis=1) for t in temps] 
sigma = [0.5**2*np.identity(p) for t in temps]                                      
tail_para = (mu[0],1**2*np.identity(p),0.1)
r = 0.15                                                                           
power = 0.1                                                                         
K = 20    


#%% AMIS sampling with parallel tempering
theta = [uniform.rvs(loc=lowbound,scale=upbound-lowbound) for t in temps]
for i in range(len(temps)):
    theta[i][4] = uniform.rvs(loc=lowbound[4],scale=Constraint_b0(theta[i]*scale,mVPD)/scale[4]-lowbound[4])
    theta[i][1] = uniform.rvs(loc=lowbound[1],scale=Constraint_p50(theta[i]*scale,mVPD)/scale[1]-lowbound[1])
logp1 = np.array(list(map(Gaussian_loglik,theta)))

sample = np.copy(theta[0]).reshape((-1,p))

## for test
sample1 = np.copy(theta[1]).reshape((-1,p))
sample2= np.copy(theta[2]).reshape((-1,p))

lik = np.array([np.copy(logp1[0])])
acc = np.zeros(temps.shape)
swapflag = np.zeros(temps.shape)


#%%
for chunckid in range(numchunck): 
    outname = outpath+sitename +'_0_'+str(chunckid).zfill(2)+'.pickle' 
    for i in range(niter):
#        print(i)
        acc = acc*(i+chunckid*niter)/(i+chunckid*niter+1)
        
        # Propose a new sample
        theta_star = [AMIS_proposal_constraint(theta[j],mu[j],sigma[j],tail_para,bounds,mVPD) for j,t in enumerate(temps)]
        
        # Evalute likelihood
        logp2 = np.array(list(map(Gaussian_loglik,theta_star))) # before tempering
        logq2 = np.array([AMIS_prop_loglik(th,mu[j],sigma[j],tail_para) for j,th in enumerate(theta_star)])
        logq1 = np.array([AMIS_prop_loglik(th,mu[j],sigma[j],tail_para) for j,th in enumerate(theta)])

        # Accept with calculated probability
        logA = (logp2-logp1)/temps-(logq2-logq1)
        accept = np.log(uniform.rvs(size=len(temps)))<logA
        acc[accept] = acc[accept]+1/(i+chunckid*niter+1)
        
        
        theta = ReplaceItems(theta,theta_star,accept)
        logp1[accept] = logp2[accept]
        
        # Swap between chains
        theta,logp1,sf = SwapChains(temps,theta,logp1)
        swapflag = np.row_stack([swapflag,sf])
    
        # Save the sample for T = 1, i.e., the target distribution
        sample = np.row_stack([sample,theta[0]]) 
        lik = np.concatenate([lik,[logp1[0]]])
        
#        # for test
        sample1 = np.row_stack([sample1,theta[1]])
        sample2 = np.row_stack([sample2,theta[2]])

        if np.mod(i,K)==0:
            rn = r/((i+1+chunckid*niter)/K)**power
            mu[0] = mu[0]+rn*np.mean(sample[-K:]-mu[0],axis=0)
            sigma[0] = sigma[0]+rn*(np.dot(np.transpose(sample[-K:]-mu[0]),sample[-K:]-mu[0])/K-sigma[0])
            mu[1] = mu[1]+rn*np.mean(sample1[-K:]-mu[1],axis=0)
            sigma[1] = sigma[1]+rn*(np.dot(np.transpose(sample1[-K:]-mu[1]),sample1[-K:]-mu[1])/K-sigma[1])
            mu[2] = mu[2]+rn*np.mean(sample2[-K:]-mu[2],axis=0)
            sigma[2] = sigma[2]+rn*(np.dot(np.transpose(sample2[-K:]-mu[2]),sample2[-K:]-mu[2])/K-sigma[2])

    
    print('Acceptance rate: '+str(acc))
    print('Swap rate:')
    print(np.sum(swapflag,axis=0)/niter)
    
    sdf = pd.DataFrame(np.column_stack([sample*scale,lik]),columns = varnames)
    sdf.to_pickle(outname)
    sample = sample[-1,:]
    lik = [lik[-1]]

##   Optional: save chains at higher temperatures
    sdf = pd.DataFrame(np.column_stack([sample1*scale]),columns = varnames[:-1])
    sdf.to_pickle(outpath+sitename +'_1_'+str(chunckid).zfill(2)+'.pickle')
    sdf = pd.DataFrame(np.column_stack([sample2*scale]),columns = varnames[:-1])
    sdf.to_pickle(outpath+sitename +'_2_'+str(chunckid).zfill(2)+'.pickle')
    sample1 = sample1[-1,:]
    sample2 = sample2[-1,:]
