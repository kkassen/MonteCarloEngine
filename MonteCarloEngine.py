# Cite: Annotated Alogrithms in Python by Di Pierro
# Cite: Dr. John McDonald, DePaul University
import numpy as np

def bootstrap(x, confidence=.68, nSamples=100):
    means = []
    for k in range(nSamples):
        sample = np.random.choice(x, size=len(x), replace=True)
        means.append(np.mean(sample))
    means.sort()
    leftTail = int(((1.0 - confidence)/2) * nSamples)
    rightTail = (nSamples - 1) - leftTail
    return means[leftTail], np.mean(x), means[rightTail]

class MonteCarlo:

    def SimulateOnce(self):
        raise NotImplementedError
        
    def SimulateOneBatch(self, batchSize):
        raise NotImplementedError
        
    def var(self, risk = .05):
        if hasattr(self, "results"): 
    	    self.results.sort()
    	    index = int(len(self.results)*risk)   
    	    return(self.results[index]) 
        else:
            print("RunSimulation must be executed before the method 'var'")
            return 0.0

    def RunSimulation(self, threshold=.001, simCount=100000):
        self.results = []       # Array to hold the results
        sum1 = 0.0              # Sum of the results
        sum2 = 0.0              # Sum of the results^2
        
        self.convergence = False
        for k in range(simCount):   
            x = self.SimulateOnce()     
            self.results.append(x)      
            sum1 += x                   
            sum2 += x*x                 
                      
            if k > 100:
                mu = float(sum1)/k                 
                var = (float(sum2)/(k-1)) - mu*mu   
                dmu = np.sqrt(var / k)              

                if (k % 1000 == 0):
                    print("k = " + str(k) + ", dmu = " + str(round(dmu, 4)))
                if dmu < abs(mu) * threshold:
                    self.convergence = True

        return bootstrap(self.results)
           
    def RunBatchedSimulation(self, batchSize=100, threshold=.001, simCount=100000):
        self.results = np.array([])     
        sum1 = 0.0    
        sum2 = 0.0         
        
        self.convergence = False
        for k in range(0, simCount, batchSize):   
            x = self.SimulateOneBatch(batchSize)          
            self.results = np.append(self.results, x)       
            sum1 += np.sum(x)                              
            sum2 += np.sum(x*x)                           
                      
            if k > 100:
                mu = float(sum1)/k                  
                var = (float(sum2)/(k-1)) - mu*mu   
                dmu = np.sqrt(var / k)             
                                                    
                if (k % 1000 == 0):
                    print("k = " + str(k) + ", dmu = " + str(round(dmu, 4)))
                if dmu < abs(mu) * threshold:
                    self.convergence = True
                
        return bootstrap(self.results)
