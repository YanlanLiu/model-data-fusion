import os, glob
import numpy as np
import pandas as pd
from scipy.stats import uniform,norm
import warnings; warnings.simplefilter("ignore")

from myFun import SoilRoot, InitialState, f_PM_hydraulics,dailyAvg
from myFun import Constraint_p50, Constraint_b0,AMIS_proposal_constraint,AMIS_prop_loglik


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

mu = np.mean(np.column_stack([lowbound,upbound]),axis=1)
sigma = 0.5**2*np.identity(p)
tail_para = (mu,1**2*np.identity(p),0.1) # mu0, sigma0, ll
r = 0.15
power = 0.1
K = 20
#%% AMIS sampling without parallel tempering
theta = uniform.rvs(loc=lowbound,scale=upbound-lowbound)
theta[4] = uniform.rvs(loc=lowbound[4],scale=Constraint_b0(theta*scale,mVPD)/scale[4]-lowbound[4])
theta[1] = uniform.rvs(loc=lowbound[1],scale=Constraint_p50(theta*scale,mVPD)/scale[1]-lowbound[1])
logp1 = Gaussian_loglik(theta)

sample = np.copy(theta).reshape((-1,p))

lik = [np.copy(logp1)]
acc = 0

for chunckid in range(numchunck): 
    outname = outpath+sitename +'_'+str(chunckid).zfill(2)+'.pickle' 
    for i in range(niter):
        acc = acc*(i+chunckid*niter)/(i+1+chunckid*niter)
        
        # Propose a new sample
        theta_star = AMIS_proposal_constraint(theta,mu,sigma,tail_para,bounds,mVPD)
        
        # Evalute likelihood
        logp2 = Gaussian_loglik(theta_star) # before tempering
        logq2 = AMIS_prop_loglik(theta_star,mu,sigma,tail_para)
        logq1 = AMIS_prop_loglik(theta,mu,sigma,tail_para)
        
        # Accept with calculated probability
        logA = (logp2-logp1)-(logq2-logq1)
        if np.log(uniform.rvs())<logA:
            acc = acc+1/(i+chunckid*niter+1)
            theta = np.copy(theta_star)
            logp1 = np.copy(logp2)
    
        # Save the sample for T = 1, i.e., the target distribution
        sample = np.row_stack([sample,theta]) 
        lik = np.concatenate([lik,[logp1]])
        
        # Update proposal distribution
        if np.mod(i,K)==0:
            rn = r/((i+1+chunckid*niter)/K)**power
            mu = mu+rn*np.mean(sample[-K:]-mu,axis=0)
            sigma = sigma+rn*(np.dot(np.transpose(sample[-K:]-mu),sample[-K:]-mu)/K-sigma)
            print(np.linalg.det(sigma))
    
    print('Acceptance rate: '+str(acc))
    
    sdf = pd.DataFrame(np.column_stack([sample*scale,lik]),columns = varnames)
    sdf.to_pickle(outname)
    sample = sample[-1,:]
    lik = [lik[-1]]

