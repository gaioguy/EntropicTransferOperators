import numpy as np
import scipy
import scipy.sparse
from pykeops.numpy import LazyTensor

class TKeopsSinkhornSolverStandard:

    MSG_EXCEEDMAXITERATIONS=30101

    def __init__(self,posX,posY,muX,muY,errorGoal,rhoX=None,rhoY=None,alpha=None,beta=None,\
                eps=None,epsInit=None,epsSteps=None,epsList=None,verbose=False):
    
    
        self.verbose=verbose
        
        self.posX=posX
        self.posY=posY
    
        self.muX=muX
        self.muY=muY
        self.errorGoal=errorGoal

        self.xres=self.posX.shape[0]
        self.yres=self.posY.shape[0]
        self.dim=self.posX.shape[1]

        
        # the reference measure for entropic regularization is given by rhoX \otimes rhoY
        # if these are not provided, muX \otimes muY is used
        if rhoX is None:
            self.rhoX=muX
        else:
            self.rhoX=rhoX

        if rhoY is None:
            self.rhoY=muY
        else:
            self.rhoY=rhoY

        # allocate new dual variables, or use the ones provided
        if alpha is None:
            self.alpha=np.zeros(self.xres,dtype=np.float32)
        else:
            self.alpha=alpha
        if beta is None:
            self.beta=np.zeros(self.yres,dtype=np.float32)
        else:
            self.beta=beta

        self.setEpsScaling(eps=eps,epsInit=epsInit,epsSteps=epsSteps,epsList=epsList)    
        # set current value of eps to None
        self.eps=None
        
        
        
        # other parameters
        self.cfg={
            "maxIterations" : 10000,
            "innerIterations" : 100,
            "truncation_thresh" : 1E-10
        }
        
        
        # setup keops lazy tensors
        self.kePosX = LazyTensor(self.posX.reshape((self.xres,1,self.dim)))
        self.kePosY = LazyTensor(self.posY.reshape((1,self.yres,self.dim)))
        self.keC=((self.kePosX-self.kePosY)**2).sum(-1)
        self.keAlpha=LazyTensor(self.alpha.reshape((self.xres,1,1)))
        self.keBeta=LazyTensor(self.beta.reshape((1,self.yres,1)))
        self.keMuXLog=np.log(self.muX).reshape((self.xres,1,1))
        self.keMuYLog=np.log(self.muY).reshape((1,self.yres,1))
        self.keRhoXLog=np.log(self.rhoX).reshape((self.xres,1,1))
        self.keRhoYLog=np.log(self.rhoY).reshape((1,self.yres,1))

    def setEpsScaling(self,eps=None,epsInit=None,epsSteps=None,epsList=None):
        # set up epsScaling
        if epsList is not None:
            self.epsList=epsList
        else:
            if eps is not None:
                if epsInit is None:
                    self.epsList=[eps]
                else:
                    if epsSteps is None:
                        # compute epsSteps such that ratio between two successive eps is bounded by 2
                        epsSteps=int((np.log(epsInit)-np.log(eps))/np.log(2))
                        if epsSteps>0:
                            epsSteps+=1
                            #epsFactor=(epsInit/eps)**(1./epsSteps)
                            self.epsList=[eps*(epsInit/eps)**(1-i/epsSteps) for i in range(epsSteps+1)]
                        elif epsSteps<0:
                            epsSteps-=1
                            #epsFactor=(epsInit/eps)**(1./epsSteps)
                            self.epsList=[eps*(epsInit/eps)**(1-i/epsSteps) for i in range(0,epsSteps-1,-1)]
                        else:
                            #epsFactor=1.
                            self.epsList=[epsInit,eps]
                    else:
                        self.epsList=[eps*(epsInit/eps)**(1-i/epsSteps) for i in range(epsSteps+1)]
            else:
                self.epsList=None

    def setRelEpsList(self,epsNew):
        epsOld=self.eps
        self.setEpsScaling(eps=epsNew,epsInit=epsOld)

    def changeEps(self,eps):
        self.eps=eps
        self.SinkhornY=(self.keAlpha-self.keC)/float(self.eps)+self.keRhoXLog
        self.SinkhornX=(self.keBeta-self.keC)/float(self.eps)+self.keRhoYLog
        self.SinkhornPi=(self.keAlpha+self.keBeta-self.keC)/float(self.eps)+self.keRhoXLog+self.keRhoYLog

        self.deltaAlpha=self.eps*(np.log(self.muX)-np.log(self.rhoX))
        self.deltaBeta=self.eps*(np.log(self.muY)-np.log(self.rhoY))                

    def solve(self):
        if self.epsList is None:
            raise ValueError("epsList is None")
        for eps in self.epsList:
            self.changeEps(eps)
            if self.verbose: print("eps: {:e}".format(self.eps))
            msg=self.solveSingle()
            if msg!=0:
                return msg
        return 0

    def solveSingle(self):
        nIterations=0

        while True:
            
            # inner iterations
            self.iterate(self.cfg["innerIterations"])
            # retrieve iteration accuracy error
            error=self.getError()

            # if self.verbose: print("\terror: {:e}".format(error))
            if error<=self.errorGoal:
                # if numerical accuracy has been achieved, finish
                return 0

            # increase iteration counter
            nIterations+=self.cfg["innerIterations"]
            if nIterations>self.cfg["maxIterations"]:
                return self.MSG_EXCEEDMAXITERATIONS
            

    ##############################################################
    # model specific methods, here for standard balanced OT
        
    def getError(self):
        # return L1 error of first marginal
        muXEff=np.exp(self.SinkhornPi.logsumexp(1)).ravel()
        return np.sum(np.abs(muXEff-self.muX))


    def iterate(self,n):
        # standard Sinkhorn iterations
        for i in range(n):
            self.alpha[:]=-self.eps*self.SinkhornX.logsumexp(1)[:,0]+self.deltaAlpha
            self.beta[:]=-self.eps*self.SinkhornY.logsumexp(0)[:,0]+self.deltaBeta


    def extractCoupling(self,thresh=1E-3,maxLen=1E6):
        # extract sparse coupling
        data=[]
        indices=[]
        indptr=[0]
        curLen=0
        for i in range(self.xres):
            piRow=np.exp((-np.sum((self.posX[i]-self.posY)**2,axis=1)+self.alpha[i]+self.beta)/self.eps)*self.rhoX[i]*self.rhoY
            piMax=np.max(piRow)
            active=np.where(piRow>piMax*thresh)[0]
            data.append(piRow[active])
            indices.append(active)
            curLen+=active.shape[0]
            indptr.append(curLen)
            if curLen>maxLen:
                raise ValueError("too many indices in matrix instantiation")

        data=np.concatenate(data)
        indices=np.concatenate(indices)
        indptr=np.array(indptr)
        piCSR=scipy.sparse.csr_matrix((data,indices,indptr),shape=(self.xres,self.yres))
        return piCSR


