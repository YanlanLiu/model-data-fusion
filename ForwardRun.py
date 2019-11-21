import numpy as np
import pandas as pd
import warnings; warnings.simplefilter("ignore")
from myFun import SoilRoot,XylemLeaf, InitialState, f_PM_hydraulics


# Site specific parameters
sitename = 'US-Me2'
soil_texture = 2; root_type = 4
root_depth = 2; canopy_height = 33; tower_height = 47 # in meters
soil_b = 0.62 # calibration coefficient for soil hydraulic parameter, unitless
boundary_cond = 0.253 # boundary condition of the second soil layer, in volumnetric water content
gpmax = 3.9e-6 # maximum xylem conductance, m/s/MPa
p50 = -5.86 # psi_50, MPa
aa = 4.53 # shape parameter of vulnerability curve, unitless
lww = 6255 # marginal water use efficiency under a well-watered condition # umol/mol
b0 = -1.28 # sensitivity of MWUE to leaf water potential, /MPa
nobsinaday = 48 # number of observations per day


#%%
df = pd.read_csv(sitename+'.csv')
SRparas = SoilRoot(soil_texture,root_type,root_depth,canopy_height,tower_height,24*3600/nobsinaday,soil_b,boundary_cond)
XLparas = XylemLeaf(gpmax,p50,aa,lww,b0)
Init = InitialState(-0.05,-0.1,-0.2,-0.1,df['SOILM'][0],df['SOILM2'][0])

ww = 0 # simulate the second layer soil moisture
#ww = 1  # use SOILM2 in df as the second layer soil moisture
ET,PSIL = f_PM_hydraulics(df,XLparas,SRparas,Init,ww)

#%%
import matplotlib.pyplot as plt
from myFun import dailyAvg

plt.figure()
plt.plot(dailyAvg(df['ET'],nobsinaday*15),'-r',label='observed')
plt.plot(dailyAvg(ET,nobsinaday*15),'-k',label='modeled')
plt.xlabel('Time')
plt.ylabel('ET (mmol/m2/s)')
plt.legend()

plt.figure()
plt.plot(dailyAvg(PSIL,nobsinaday*15),'-k')
plt.xlabel('Time')
plt.ylabel('Leaf water potential (MPa)')

plt.figure()
plt.plot(PSIL)
