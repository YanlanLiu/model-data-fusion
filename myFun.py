import numpy as np
import time
from scipy.stats import  multivariate_normal,bernoulli,uniform

# Clapp and Hornberger, 1978, WRR
BB = np.array([4.05,4.38,4.90,5.3,5.39,7.12,7.75,8.52,10.4,10.4,11.4])
PSAT = -np.array([12.1,9.0,21.8,78.6,47.8,29.9,35.6,63.0,15.3,49,40.5])*1e-2*1e3*9.8*1e-6
THETAS = np.array([0.395,0.410,0.435,0.485,0.451,0.420,0.477,0.476,0.426,0.492,0.482])
KSAT = np.array([1.056,0.938,0.208,0.0432,0.0417,0.0378,0.0102,0.0147,0.0130,0.0062,0.0077])*1e-2/60

def ClappHornberger(pct,th,calib):
    if type(pct)!=int: # when input is soil texture
        sand,silt,clay = pct
        if silt+1.5*clay<15: 
            tx = 0 # sand
        elif silt+1.5*clay>=15 and silt+2*clay<30: 
            tx = 1 # loamy sand
        elif (7<=clay<20 and sand>52 and silt+2*clay>=30) or (clay<7 and silt<50 and silt+2*clay>=30):
            tx = 2 # sandy loam
        elif (silt>=50 and 12<=clay<27) or (50<=silt<80 and clay<12):
            tx = 3 # silt loam
        elif 7<=clay<27 and 28<=silt<50 and sand <=52:
            tx = 4 # loam
        elif 20<=clay<35 and silt<28 and sand >45:
            tx = 5 # sandy clay loam
        elif 27<=clay<40 and sand<=20:
            tx = 6 # silt clay loam
        elif 27<=clay<40 and 20<sand<=45:
            tx = 7 # clay loam
        elif clay>=35 and sand>45:
            tx = 8 # sandy clay
        elif clay>=40 and silt>=40:
            tx = 9 # silt clay
        elif clay>=40 and sand<=45 and silt<40:
            tx = 10 # clay  
        elif silt>=80 and clay<12: # silt
            tx = 3 # silt
        else:
            tx = np.nan
    else: # when input is soil texture
        tx = int(np.copy(pct))
    
    b = BB[tx]*calib
    ssat = THETAS[tx]
    s = min(max(th/ssat,0),1)
    psis = PSAT[tx]*(s)**(-b)
    k = KSAT[tx]*(s)**(2*b+3)
    sfc = (-0.01/PSAT[tx])**(-1/b)*ssat
    sw = (-3/PSAT[tx])**(-1/b)*ssat
    sh = (-10/PSAT[tx])**(-1/b)*ssat
    return k,psis,ssat,b,sfc,sw,sh


# (Jackson et al., 1996, Oecologia; Jackson et al., 1997, PNAS)
RAIlist = [4.6,10,5.5,11.6,11.0,9.8,79.1,6.3,7.4,42.5,5.2]
betalist = [0.943,0.961,0.970,0.950,0.980,0.967,0.943,0.982,0.972,0.972,0.909]

class SoilRoot:
    def __init__(self,tx,root_type,root_depth,canopy_height,tower_height,dt,calib,bc):
        self.tx = int(tx)
        self.calib = np.copy(calib)
        self.bc = np.copy(bc)
        tmp,tmp,self.ssat1,tmp,self.sfc1,self.sw1,self.sh1 = ClappHornberger(self.tx,1,self.calib)
        tmp,tmp,self.ssat2,tmp,self.sfc2,self.sw2,self.sh2 = ClappHornberger(self.tx,1,self.calib)
        self.Zr1 = min(0.3,root_depth)
        self.Zr2 = max(0,root_depth-0.3)
        self.RAI1 = RAIlist[root_type]*(1-(self.Zr2>0)*betalist[root_type]**30)
        self.RAI2 = RAIlist[root_type]-self.RAI1
        self.ch = canopy_height 
        self.z = tower_height 
        self.dt = dt
        
        
class XylemLeaf:
    def __init__(self,gpmax,p50,aa,lww,b0):
        self.gpmax = gpmax
        self.p50 = p50
        self.aa = aa
        self.lww = lww
        self.b0 = b0

class InitialState:
    def __init__(self,psir,psix,psil,pisl_avg,s1,s2):
        self.psir = psir
        self.psix = psix
        self.psil = psil
        self.psil_avg = pisl_avg
        self.s1 = s1
        self.s2 = s2

class Climate:
    def __init__(self,temp,vpd,rnet,lai):
        self.temp = temp
        self.vpd = vpd
        self.rnet = rnet
        self.lai = lai

# constants and photosynthesis parameters
rhow = 1e3; g = 9.81
a0 = 1.6; 
UNIT_1 = a0*18*1e-6 # mol CO2 /m2/s -> m/s, H2O
UNIT_2 = 1e6 # Pa -> MPa
UNIT_3 = 273.15 # Degree C -> K
p0 = 101325 # Pa
R = 8.31*1e-3 # Gas constant, kJ/mol/K
hc = 2e-25 # Planck constant times light speed, J*s times m/s
wavelen = 500e-9 # wavelength of light, m
Ephoton = hc/wavelen # energy of photon, J
NA = 6.02e23 # Avogadro's constant, /mol
koptj = 155.76 #  umol/m2/s
Haj = 43.79 # kJ/mol
Hdj = 200; # kJ/mol
Toptj = 32.19+UNIT_3 # K
koptv = 174.33 # umol/m2/s
Hav = 61.21 # kJ/mol
Hdv = 200 # kJ/mol
Toptv = 37.74+UNIT_3 # K
Coa = 210 # mmol/mol
kai1 = 0.9
kai2 = 0.3
ca = 400; ca0 = 400
karman = 0.4
rhohat = 44.6 # mol/m3
Cpmol = 1005*28.97*1e-3 # J/kg/K*kg/mol -> J/mol/K
lambdamol = 40660 # J/mol


def LightExtinction(DOY,lat,x):
    B = (DOY-81)*2*np.pi/365
    ET = 9.87*np.sin(2*B)-7.53*np.cos(B)-1.5*np.sin(B)
    DA = 23.45*np.sin((284+DOY)*2*np.pi/365)# Deviation angle
    LST = np.mod(DOY*24*60,24*60)
    AST = LST+ET
    h = (AST-12*60)/4 # hour angle
    alpha = np.arcsin((np.sin(np.pi/180*lat)*np.sin(np.pi/180*DA)+np.cos(np.pi/180*lat)*np.cos(np.pi/180.*DA)*np.cos(np.pi/180*h)))*180/np.pi # solar altitude
    zenith_angle = 90-alpha
    Vegk = np.sqrt(x**2+np.tan(zenith_angle/180*np.pi)**2)/(x+1.774*(1+1.182)**(-0.733)) # Campbell and Norman 1998
    return Vegk

def dailyAvg(data,windowsize):
    data = np.array(data)
    data = data[0:windowsize*int(len(data)/windowsize)]
    return np.nanmean(np.reshape(data,[int(len(data)/windowsize),windowsize]),axis=1)


T2ES  = lambda x: 0.6108*np.exp(17.27*(x-UNIT_3)/(x-UNIT_3+237.3))# saturated water pressure, kPa
def PenmanMonteith(temp,rnet,vpd,gv,gh): # vpd in mol/mol
    Delta = 4098*T2ES(temp)*1e3/(237.3+temp-UNIT_3)**2 # Pa/K
    E = (Delta*rnet+p0*Cpmol*gh*vpd)/(Delta*lambdamol+p0*Cpmol*gh/gv)
    return E*(rnet>0)

#%%

DPSI = 0.01
PSIL_RANGE = np.arange(0,-15,-DPSI)

def f_PM_hydraulics(df,XLparas,SRparas,Init,ww):
    RNg = df['RNET']*np.exp(-df['LAI']*df['Vegk'])
    RNl = df['RNET']-RNg
    
    ggh = np.array(df['GA_U'])
    ggv = np.array(1/(1/ggh+1/df['GSOIL']))
    Eg = PenmanMonteith(df['TEMP'],RNg,df['VPD'],ggv,ggh) # mol/m2/s
    
    PSIL = np.zeros([len(df),])+Init.psil
    El = np.zeros([len(df),])
    
    VC = VulnerabilityCurve(PSIL_RANGE,XLparas.gpmax,XLparas.p50,XLparas.aa)
    KS = np.cumsum(VC*DPSI)    
    for t in range(1,len(df)):
        Clm = Climate(df['TEMP'][t],df['VPD'][t],RNl[t],df['LAI'][t])
        El[t],PSIL[t],psir,s2 = calSPAC(Clm,df['GA'][t],SRparas,XLparas,KS,Init,ww)
        if t>47:psil_avg = np.mean(PSIL[t-47:t+1])
        else: psil_avg = Init.psil_avg
        if ww==0:
            Init = InitialState(psir,Init.psix,PSIL[t],psil_avg,df['SOILM'][t],s2)
        else:
            Init = InitialState(psir,Init.psix,PSIL[t],psil_avg,df['SOILM'][t],df['SOILM2'][t])
    return (El+Eg)*1e3,PSIL

def calSPAC(Clm,glh,SRparas,XLparas,KS,Init,ww):
    ll = MWU(XLparas.lww,XLparas.b0,ca,Init.psil_avg)
    gs, An, H, flag, ci, a1, a2 = f_carbon(Clm,ll) # mol H2O /m2/s
    gs = gs*a0*Clm.lai
    glv = 1/(1/glh+1/gs)
    E = PenmanMonteith(Clm.temp,Clm.rnet,Clm.vpd,glv,glh) # mol/m2/s
#    Tl = (Clm.rnet-lambdamol*E)/glh/Cpmol+Clm.temp
    Ems = E*18*1e-6 # m/s
    psis,gsr,s2 = soilhydro(Init,SRparas,ww)
    psir = psis-Ems/gsr
    if psir<min(PSIL_RANGE):psir=min(PSIL_RANGE)
    tmp = np.abs(KS-KS[int(-psir/DPSI)]-Ems) # \int_{\psi_r}^{\psi_l}KS(x)=Ems
    psil = PSIL_RANGE[tmp==min(tmp)]
    if psil==min(PSIL_RANGE):E = 0
    return E,psil,psir,s2

def VulnerabilityCurve(psil,gpmax,p50,aa):
    return gpmax/(1+(psil/p50)**aa)

def MWU(lww,b0,ca,psil):
    return ca/ca0*lww*np.exp(b0*psil)

def f_carbon(Clm,ll):
#    ll = np.array([ll])
    T,RNET,VPD = (Clm.temp,Clm.rnet,Clm.vpd)
    
    PAR = RNET/(Ephoton*NA)*1e6 # absorbed photon irradiance, umol photons /m2/s, PAR
    
    Vcmax = koptv*Hdv*np.exp(Hav*(T-Toptv)/T/R/Toptv)/(Hdv-Hav*(1-np.exp(Hav*(T-Toptv)/T/R/Toptv))) # umol/m2/s
    Jmax = koptj*Hdj*np.exp(Haj*(T-Toptj)/T/R/Toptj)/(Hdj-Haj*(1-np.exp(Haj*(T-Toptj)/T/R/Toptj)))
    TC = T-UNIT_3 # C
    Kc = 300*np.exp(0.074*(TC-25)) # umol/mol
    Ko = 300*np.exp(0.015*(TC-25)) # mmol/mol
    cp = 36.9+1.18*(TC-25)+0.036*(TC-25)**2
    J = (kai2*PAR+Jmax-np.sqrt((kai2*PAR+Jmax)**2-4*kai1*kai2*PAR*Jmax))/2/kai1 # umol electrons /m2/s
    Rd = 0.015*Vcmax

    a1 = J/4;a2 = 2*cp # Rubisco limited photosynthesis
    tmpsqrt = a0*VPD*ll*a1**2*(ca-cp)*(a2+cp)*(a2+ca-2*a0*VPD*ll)**2*(a2+ca-a0*VPD*ll)
    if tmpsqrt>0:
        gc1 = -a1*(a2-ca+2*cp)/(a2+ca)**2+np.sqrt(tmpsqrt)/(a0*VPD*ll*(a2+ca)**2*(a2+ca-a0*VPD*ll))
        A = -gc1
        B = gc1*ca-a2*gc1-a1+Rd
        C = ca*a2*gc1+a1*cp+a2*Rd
        ci1 = (-B-np.sqrt(B**2-4*A*C))/(2*A)
        An1 = gc1*(ca-ci1)
        if np.isnan(An1) or An1<0 or gc1<0: gc1, ci1, An1 = (0,0,0)
    else:
        gc1, ci1, An1 = (0,0,0)
        
    a1 = Vcmax;a2 = Kc*(1+Coa/Ko) # RuBP limited photosynthesis
    tmpsqrt = a0*VPD*ll*a1**2*(ca-cp)*(a2+cp)*(a2+ca-2*a0*VPD*ll)**2*(a2+ca-a0*VPD*ll)
    if tmpsqrt>0:
        gc2 = -a1*(a2-ca+2*cp)/(a2+ca)**2+np.sqrt(tmpsqrt)/(a0*VPD*ll*(a2+ca)**2*(a2+ca-a0*VPD*ll))
        A = -gc2
        B = gc2*ca-a2*gc2-a1+Rd
        C = ca*a2*gc2+a1*cp+a2*Rd
        ci2 = (-B-np.sqrt(B**2-4*A*C))/(2*A)
        An2 = gc2*(ca-ci2)
        if np.isnan(An2) or An2<0 or gc2<0: gc2, ci2, An2 = (0,0,0)
    else:
        gc2, ci2, An2 = (0,0,0)
        
    flag = (An1<=An2)*1
    An = min(An1,An2)
    gc = gc1*flag+gc2*(1-flag)
    ci = ci1*flag+ci2*(1-flag)
    H = gc*(ca-ci)-a0*gc*VPD*ll
    a1 = J/4*flag+Vcmax*(1-flag)
    a2 = 2*cp*flag+Kc*(1+Coa/Ko)*(1-flag)
    return gc,An,H,flag,ci,a1,a2


def soilhydro(Init,SRparas,ww):
    K1,P1,tmp,tmp,tmp,tmp,tmp = ClappHornberger(SRparas.tx,Init.s1,SRparas.calib)
    gsr1 = K1*np.sqrt(SRparas.RAI1)/(rhow*g*SRparas.Zr1*np.pi)*UNIT_2
    if SRparas.Zr2>0:
        K2,P2,tmp,tmp,tmp,tmp,tmp = ClappHornberger(SRparas.tx,Init.s2,SRparas.calib)
        gsr2 = K2*np.sqrt(SRparas.RAI2)/(rhow*g*SRparas.Zr2*np.pi)*UNIT_2
        gsr = gsr1+gsr2
        psis = (gsr1*P1+gsr2*P2)/gsr
        if ww==0: # model the second layer soil moisture
            E2 = gsr2*(P2-Init.psir)
            K12 = 2/(1/K1+1/K2)
            L12 = K12*(P1+(SRparas.Zr1+SRparas.Zr2)/2*rhow*g/UNIT_2-P2)/(rhow*g*(SRparas.Zr1+SRparas.Zr2)/2/UNIT_2) # m/s
            if SRparas.bc>0:
                K3,P3,tmp,tmp,tmp,tmp,tmp = ClappHornberger(SRparas.tx,SRparas.bc,SRparas.calib)
                K23 = 2/(1/K2+1/K3)
                L23 = K23*(P2+SRparas.Zr2/2*rhow*g/UNIT_2-P3)/(rhow*g*SRparas.Zr2/2/UNIT_2) # m/s
            else:
                L23 = 0
            s2 = min(max(Init.s2+(L12-L23-E2)/SRparas.Zr2*SRparas.dt,SRparas.sw2),SRparas.sfc2)
        else: # use data second layer soil moisture data
            s2 = 0
    else:
        psis = np.copy(P1)
        gsr = np.copy(gsr1)
        s2 = 0
    return psis,gsr,s2


def Constraint_p50(theta,mVPD): #p50 < -2/b0*np.log((1+(a0*lww*mVPD/ca)**0.5)/2)
    return -2/theta[4]*np.log((1+(a0*theta[3]*mVPD/ca)**0.5)/2)

def Constraint_b0(theta,mVPD): #b0<-1/4*np.log(ca/a0/mVPD/lww)
    return -1/4*np.log(ca/a0/mVPD/theta[3])

MAX_STEP_TIME = 10 # sec
def AMIS_proposal_constraint(theta,mu,sigma,tail_para,bounds,mVPD):
    lowbound, upbound,scale = bounds
    mu0,sigma0,ll = tail_para
    startTime_for_tictoc = time.time()
    while True:
#        bn = bernoulli.rvs(ll,size=1)
#        theta_star = multivariate_normal.rvs(mu,sigma,size = 1)*(1-bn)+multivariate_normal.rvs(mu0,sigma0,size = 1)*bn
        if bernoulli.rvs(ll,size=1) ==1:
            theta_star = multivariate_normal.rvs(mu0,sigma0,size = 1)
        else:
            theta_star = multivariate_normal.rvs(mu,sigma,size = 1)
        upbound[1] = Constraint_p50(theta_star*scale,mVPD)/scale[1]
        upbound[4] = Constraint_b0(theta_star*scale,mVPD)/scale[4]
        
        outbound = len([i for i,im in enumerate(theta_star) if ((lowbound[i]>im) or (im>upbound[i]))])
        if (outbound==0): 
            break
        if (time.time() - startTime_for_tictoc)>MAX_STEP_TIME:
            theta_star = np.copy(theta)
            break  
    return theta_star

def AMIS_prop_loglik(theta,mu,sigma,tail_para):
    mu0,sigma0,ll = tail_para
    return np.log(multivariate_normal.pdf(theta,mu0,sigma0)*ll+multivariate_normal.pdf(theta,mu,sigma)*(1-ll))

def ReplaceItems(theta,theta_star,accept):
    for i,ac in enumerate(accept):
        if ac:theta[i] = theta_star[i]
    return theta

def SwapChains(temps,theta,logp1):
    swapflag = np.zeros(temps.shape)
    swap = np.random.choice(range(len(temps)), size=2, replace=False)
    logA = (1/temps[swap[1]]-1/(temps[swap[0]]))*(logp1[swap[0]]-logp1[swap[1]])
    if np.log(uniform.rvs())<logA:
        logp_tmp = np.copy(logp1)
        logp1[swap[0]] = logp_tmp[swap[1]]
        logp1[swap[1]] = logp_tmp[swap[0]]
        theta_tmp = np.copy(theta)
        theta[swap[0]] = theta_tmp[swap[1]]
        theta[swap[1]] = theta_tmp[swap[0]] 
        swapflag[swap] = 1
    return theta,logp1,swapflag
