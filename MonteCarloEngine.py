# Cite:  Annotated Alogrithms in Python by Di Pierro
# Cite: Dr. John McDonald, DePaul University
import numpy as np

# A function that helps us compute the statistics for the MonteCarlo simulation.
# Notice that the default confidence interval is the 68th percentile which is
# equivalent to one standard-deviation if the distribution is sufficiently close
# to normal.
def bootstrap(x, confidence=.68, nSamples=100):
    # Make "nSamples" new datasets by re-sampling x with replacement
    # the size of the samples should be the same as x itself
    means = []
    for k in range(nSamples):
        sample = np.random.choice(x, size=len(x), replace=True)
        means.append(np.mean(sample))
    means.sort()
    leftTail = int(((1.0 - confidence)/2) * nSamples)
    rightTail = (nSamples - 1) - leftTail
    return means[leftTail], np.mean(x), means[rightTail]

# A general MonteCarlo engine that runs a simulation many times and computes the average 
# and the error in the average (confidence interval for a certain level). 
class MonteCarlo:
    """
    The SimulateOnce method is declared as abstract (it doesn't have an implementation
    in the base class) that must be extended/overriden to build the simualtion.
    """
    
    # Will return a single number
    def SimulateOnce(self):
        raise NotImplementedError
        
    # Will return an array of "batchSize" numbers
    def SimulateOneBatch(self, batchSize):
        raise NotImplementedError
        
    # This function computes the value at risk amount for the results of a simulation
    # It can only be run after the main simulation is run, so in this function we
    # test for whether the results member exists and if not, print an error    
    def var(self, risk = .05):
        if hasattr(self, "results"): 
    	    self.results.sort()
    	    index = int(len(self.results)*risk)   
    	    return(self.results[index]) 
        else:
            print("RunSimulation must be executed before the method 'var'")
            return 0.0

    # This function runs the simulation.  Note that it stores the results of each of the
    # trials in an array that is a CLASS variable, not a local variable in this function
    # so, we can get this array from the MonteCarlo object after running if we wish.
    def RunSimulation(self, threshold=.001, simCount=100000):
        self.results = []       # Array to hold the results
        sum1 = 0.0              # Sum of the results
        sum2 = 0.0              # Sum of the results^2
        
        # Now, we set up the simulation loop
        self.convergence = False
        for k in range(simCount):   
            x = self.SimulateOnce()     # Run the simulation
            self.results.append(x)      # Add the result to the array
            sum1 += x                   # Add it to the sum
            sum2 += x*x                 # Add the square to the sum of squares
            
            # Go to at least a 100 cycles before testing for convergence.             
            if k > 100:
                mu = float(sum1)/k                  # Compute the mean
                var = (float(sum2)/(k-1)) - mu*mu   # An alternate calculation of the variance
                dmu = np.sqrt(var / k)              # Standard error of the mean
                                                    
                # If the estimate of the error in mu is within "threshold" percent
                # then set convergence to true.  We could also break out early at this 
                # point if we wanted to 
                if (k % 1000 == 0):
                    print("k = " + str(k) + ", dmu = " + str(round(dmu, 4)))
                if dmu < abs(mu) * threshold:
                    self.convergence = True

        # Bootstrap the results and return not only the mean, but the confidence interval
        # as well.  [mean - se, mean, mean + se]
        return bootstrap(self.results)

    # This function runs a set of batched simulations, calling "SimulateOneBatch"
    # each time.  The results are all pooled and processed in the same way as before
    # but if "SimulateOneBatch" is properly implemented with array operations it is 
    # potentially much faster than the single pass alternative!                    
    def RunBatchedSimulation(self, batchSize=100, threshold=.001, simCount=100000):
        self.results = np.array([])       # Array to hold the results
        sum1 = 0.0              # Sum of the results
        sum2 = 0.0              # Sum of the results^2
        
        # Now, we set up the simulation loop
        self.convergence = False
        for k in range(0, simCount, batchSize):   
            x = self.SimulateOneBatch(batchSize)            # Run the simulation
            self.results = np.append(self.results, x)       # Add the list of result to the array
            sum1 += np.sum(x)                               # Add the sum to the total sum
            sum2 += np.sum(x*x)                             # Add the sum of the squares to the total sum of squares
            
            # Go to at least a 100 cycles before testing for convergence.             
            if k > 100:
                mu = float(sum1)/k                  # Compute the mean
                var = (float(sum2)/(k-1)) - mu*mu   # An alternate calculation of the variance
                dmu = np.sqrt(var / k)              # Standard error of the mean
                                                    
                # If the estimate of the error in mu is within "threshold" percent
                # then set convergence to true.  We could also break out early at this 
                # point if we wanted to 
                if (k % 1000 == 0):
                    print("k = " + str(k) + ", dmu = " + str(round(dmu, 4)))
                if dmu < abs(mu) * threshold:
                    self.convergence = True
                
        # Bootstrap the results and return not only the mean, but the confidence interval
        # as well.  [mean - se, mean, mean + se]
        return bootstrap(self.results)

            