# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:10:56 2023

@author: Tom Ferguson, PhD, University of Alberta
"""


def AS_KFSMP(parameters,
            rewardVal,           
            trialParam,
            blockType):
    
    import numpy as np

    # Initialize Parameters
    numArms = int(trialParam[0])
    numBlocks = int(trialParam[1])
    numTrials = int(trialParam[2])
    
    # Parameters
    sigXi = 1
    sigEps = parameters[0]
    temperature = parameters[1]
    discount = parameters[2]
    
    # Set up arrays
    selectionMat = np.zeros([numBlocks, numTrials])
    rewArray = np.zeros(shape=[numBlocks, numTrials])
    ktArray= np.zeros(shape=[numBlocks, numTrials, numArms])

    # Overal Probability (for saving?)
    probOv = np.zeros(shape=[numBlocks, numTrials, numArms])

    # Selection for Practice Trials
    selectStart = np.zeros(shape=8)
    selectStart[0:2] = 0
    selectStart[2:4] = 1
    selectStart[4:6] = 2
    selectStart[6:8] = 3
    
    # Initialize Values?
    if blockType == 1:
        mu0 = 10
        v0 = 20
    else:
        mu0 = -10
        v0 = 20
    

    m = np.zeros(shape=[numBlocks, numTrials+1, numArms]) + mu0
    v = np.zeros(shape=[numBlocks, numTrials+1, numArms]) + v0

    for block in range(numBlocks):
        

        for trial in range(numTrials):
            
            softmaxResult = np.zeros(shape=numArms) #+.25

            selection = []
            reward = []
            
            if trial <= 7:
                                
                selection = int(selectStart[trial])
                
                # Reward
                reward = rewardVal[block, trial, selection] #/ 100
                                
                # Zero Kalman Gain Set up
                kt = np.zeros(shape=numArms)
                
                # Zero for arrays
                softmaxResult = np.zeros(shape=numArms)                

                # Zero Kalman Gain Set up
                kt = np.zeros(shape=numArms)
                
                # Calculate kalman Gain
                kt[selection] = (v[block, trial, selection] + sigXi) / (v[block, trial, selection] + sigXi + sigEps)
                
                # Update Mean and Variances
                m[block, trial+1] = m[block, trial] + kt*(reward - m[block, trial,])
                v[block, trial+1] = (1 - kt) * (v[block, trial,] + sigXi)
            
            elif trial == 8:
                
                # Indicator fun
                indi = np.zeros(shape=numArms)
                #indi[int(selectionMat[block, trial-1])] = 1
                
                # Choice stickiness
                disVal = np.multiply(indi, discount)
                        
                qValue = m[block, trial, :] / 100
                
                #Compute Softmax values
                num = np.exp(np.multiply(qValue+disVal,temperature))
                
                denom = sum(np.exp(np.multiply(qValue+disVal,temperature)));
                            
                #Find softmax result
                softmaxResult = num/denom
                
                #Find cumulative sum
                softmaxSum = np.cumsum(softmaxResult)
                
                #Assign Values to softmax options
                softmaxOptions = softmaxSum > np.random.rand()
                
                # #Find arm choice
                # selection = np.argmax(softmaxOptions)
                selection = np.argmax(m[block, trial, :])
                
                # Reward
                reward = rewardVal[block, trial, selection] #/ 100
                
                # Zero Kalman Gain Set up
                kt = np.zeros(shape=numArms)
                
                # Calculate kalman Gain
                kt[selection] = (v[block, trial, selection] + sigXi) / (v[block, trial, selection] + sigXi + sigEps)
                
                # Update Mean and Variances
                m[block, trial+1] = m[block, trial] + kt*(reward - m[block, trial,])
                v[block, trial+1] = (1 - kt) * (v[block, trial,] + sigXi)    
            
            else:
                
                # Indicator fun
                indi = np.zeros(shape=numArms)
                indi[int(selectionMat[block, trial-1])] = 1
                
                # Choice stickiness
                disVal = np.multiply(indi, discount)
                        
                qValue = m[block, trial, :] / 100
                
                #Compute Softmax values
                num = np.exp(np.multiply(qValue+disVal,temperature))
                
                denom = sum(np.exp(np.multiply(qValue+disVal,temperature)));
                            
                #Find softmax result
                softmaxResult = num/denom
                
                #Find cumulative sum
                softmaxSum = np.cumsum(softmaxResult)
                
                #Assign Values to softmax options
                softmaxOptions = softmaxSum > np.random.rand()
                
                # Find Max Value
                selection = np.argmax(softmaxOptions)

                # Reward
                reward = rewardVal[block, trial, selection] #/ 100
            
                # Zero Kalman Gain Set up
                kt = np.zeros(shape=numArms)
                
                # Calculate kalman Gain
                kt[selection] = (v[block, trial, selection] + sigXi) / (v[block, trial, selection] + sigXi + sigEps)
                
                # Update Mean and Variances
                m[block, trial+1] = m[block, trial] + kt*(reward - m[block, trial,])
                v[block, trial+1] = (1 - kt) * (v[block, trial,] + sigXi)
                
            # Save Probabilities
            probOv[block, trial, :] = softmaxResult
            
            # Assign Choice
            selectionMat[block, trial] = selection
            rewArray[block, trial] = reward
            
            ktArray[block, trial, :] = kt
            
    return selectionMat, rewArray, ktArray, probOv, m, v