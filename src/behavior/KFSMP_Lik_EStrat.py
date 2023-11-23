# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:58:14 2023

@author: Tom Ferguson, PhD, University of Alberta
"""


def LL_KFSMP(parameters,
            rewardVal,
            choices,
            trialParam,
            blockType):
    
    import numpy as np

    # Initialize Parameters
    numArms = int(trialParam[0])
    numBlocks = int(trialParam[1])
    numTrials = int(trialParam[2])
    
    # # Isolate Parameters?
    if blockType == 1:
        mu0 = 10
        v0 = 20
    else:
        mu0 = 0
        v0 = 20
    
    # mu0 = 0
    # v0 = 30
    
    sigXi_LL = 1
    sigEps_LL = parameters[0]
    temp_LL = parameters[1]
    discount_LL = parameters[2]
    
    numArms = trialParam[0]
    numBlocks = trialParam[1]
    numTrials = trialParam[2]

    # Initialize Liklihood array
    liklihoodArray = np.zeros(shape=[numBlocks, numTrials])
    
    # Extract Posterior means and variances for KF calculation
    m = np.zeros(shape=[numBlocks, numTrials+1, numArms]) + mu0
    v = np.zeros(shape=[numBlocks, numTrials+1, numArms]) + v0
        
    for block in range(numBlocks):
                
        # Loop around trials
        for trial in range(numTrials):
            
            # Find selection and reward
            selection = int(choices[block, trial])

            if trial < 8:
                                
                liklihoodArray[block, trial] = .25
    
            elif  selection == -1:
            
            # if selection == -1:
                
                liklihoodArray[block, trial] = 1
                
                # Calculate kalman Gain
                #kt[selection] = (v[bCt, tCt, selection] + sigXi) / (v[bCt, tCt, selection] + sigXi + sigEps)
                            
                # Update Mean and Variances
                m[block, trial+1] = m[block, trial]
                v[block, trial+1] = v[block, trial] + sigXi_LL
                
            else:

                reward = rewardVal[block, trial]
                
                # Indicator fun
                if trial == 0:
                    indi = np.zeros(shape=numArms)
                else:
                    indi = np.zeros(shape=numArms)
                    indi[int(choices[block, trial-1])] = 1
                
                # Choice stickiness
                disVal = np.multiply(indi, discount_LL)

                #Find Prob for sampling          
                qValue = m[block, trial, :] / 100

                #Compute Softmax values
                num = np.exp(np.multiply(qValue+disVal, temp_LL))
                denom = sum(np.exp(np.multiply(qValue+disVal, temp_LL)))
                
                #Find softmax result
                softmaxResult = num/denom
                            
                # Zero Kalman Gain Set up
                kt = np.zeros(shape=numArms)
                
                # Calculate kalman Gain
                kt[selection] = (v[block, trial, selection] + sigXi_LL) / (v[block, trial, selection] + sigXi_LL + sigEps_LL)
                            
                # Update Mean and Variances
                m[block, trial+1] = m[block, trial] + kt*(reward - m[block, trial])
                v[block, trial+1] = (1 - kt) * (v[block, trial] + sigXi_LL)
                            
                
                if trial < 8:
                    
                    liklihoodArray[block, trial] = .25
                    
                else:
                
                    liklihoodArray[block, trial] = softmaxResult[selection]
    
    #if liklihoodArray <= 0:
    liklihoodArray[liklihoodArray <= 0] = 1e-300
    
    # Deal with NaNs
    liklihoodArray[np.isnan(liklihoodArray)] = 1e-300
    
    #print([sigXi_LL, sigEps_LL])
    
    #Update Liklihood Values
    liklihoodSum = -np.sum(np.log(liklihoodArray[:, :]))
                            
    return liklihoodSum