'''
Created on 09-Apr-2015

@author: niloygupta
'''

from pyspark import SparkContext,SparkConf
import sys

from scipy.sparse import csr_matrix
import numpy as np
import math
import random

beta_value = 0.0
lambda_value = 0.0


# Logic for extracting the stratum id given userid, movie id and block size.
#First part of the equation gets the stratum id, while the second part gets the number of rotations/shifts
# Take modulus by number of workers to convert negative ids to fit within 0 to numWorkers range
def createStratum(x, numWorkers, userblock_size,movieblock_size):

    userId, movieId = x[0], x[1]
    stratumId = ((movieId/movieblock_size) - (userId/userblock_size))%numWorkers
    #stratumId = stratumId%numWorkers

    return stratumId



#Convert the csv data file into RDD objects. Each stratum is stored as a partition.
def load_data(inputV_filepath,num_factors,num_workers,sc):
    data = sc.textFile(inputV_filepath).map(lambda x: x.split(",")).map(lambda x: (int(x[0])-1, int(x[1])-1, float(x[2])))
    
    maxUserId = data.map(lambda x: x[0]).max() + 1
    maxMovieId = data.map(lambda x: x[1]).max() + 1
        
    adjRow = maxUserId/num_workers
    adjCol = maxMovieId/num_workers 
    if(maxUserId%num_workers !=0):
        adjRow = adjRow + 1
    if(maxMovieId%num_workers !=0):
        adjCol = adjCol + 1   
    block_size = (adjRow, adjCol) 

    partitions = data.keyBy(lambda x: createStratum(x, num_workers, block_size[0],block_size[1]))
    W = np.random.random_sample((maxUserId, num_factors))
    H = np.random.random_sample((num_factors, maxMovieId))
    
    partitions.cache()
    
    return partitions,W,H,block_size


#Convert each block into a sparse matrix for fast computation
def buildSparseMatrix(blockInfo,rowSize,colSize,blockEntryRow,blockEntryCol):
    rows = []
    cols = []
    ratings = []
    for triple in blockInfo:
        rows.append(triple[1][0]-blockEntryRow)
        cols.append(triple[1][1]-blockEntryCol)
        ratings.append(triple[1][2])
        #ratings.append(-1 if triple[1][2]==0 else triple[1][2])
    
    return csr_matrix((ratings, (rows,cols)), shape=(rowSize, colSize))

#Update W and H till they converge
def updateGradient(block):
    blockInfo, W, H, mbi, blockIndex,blockEntryRow,blockEntryCol = block[0], block[1], block[2], block[3], block[4],block[5],block[6]

    V = buildSparseMatrix(blockInfo,W.shape[0],H.shape[1],blockEntryRow,blockEntryCol)

    rows,cols = V.nonzero()
    count = 0
    oldLoss = 0
    newLoss = 1

    while math.fabs(newLoss -oldLoss )>1e-5 and len(rows)>0:
        oldLoss = newLoss
        k = random.randint(0,(rows).size-1)
        r = rows[k]
        c = cols[k]
        lr = math.pow((100 + mbi+count),-beta_value)
        count = count + 1
        V_val = V[r,c]
        '''if V_val == -1:
            V_val = 0.0;'''
        
        Wgrad = -2*(V_val - W[r,:].dot(H[:,c]))*(H[:,c].T) + (2*lambda_value*W[r,:])/(V[r,:].nonzero()[0].size)
        Hgrad = -2*(V_val - W[r,:].dot(H[:,c]))*(W[r,:].T) + (2*lambda_value*H[:,c])/(V[:,c].nonzero()[0].size)
        H[:,c] = H[:,c] - lr*Hgrad
        W[r,:] = W[r,:] - lr*Wgrad
        
        #V[V==-1] =  0
        P = W.dot(H)
        newLoss = l2Loss(V[rows,cols],P[rows,cols],W,H,lambda_value)
    return (blockIndex, mbi+count, W,H)
    
    
def l2Loss(V,P,W,H,lambda_value):
    l = V- P
    l = np.multiply(l,l) 
    return np.sum(l) +lambda_value*(np.sum(np.multiply(W,W)) + np.sum(np.multiply(H,H)))
      



def factorize_matrix(partitions,W,H,block_size,T,num_workers,sc):
    
    adjRow = block_size[0]
    adjCol = block_size[1]
    maxMovieId = H.shape[1]
    maxUserId = W.shape[0]

    stratumIndices = {}
    
    # Store block ranges for each stratum
    for stratum in xrange(num_workers):                     
        blockRanges = [] 
        for b in xrange(num_workers):                  
            blockRowIndex = np.array([b*adjRow,(b+1)*adjRow])
            blockColIndex = np.array([(b+stratum)*adjCol,(b+stratum+1)*adjCol])
            
            if(blockColIndex[0] > maxMovieId):
                blockColIndex[0] = blockColIndex[0]%maxMovieId - 1
            
            blockColIndex[1] = blockColIndex[0] + adjCol
            
            if(blockColIndex[1]>maxMovieId):
                blockColIndex[1] = maxMovieId
                
            if(blockRowIndex[1]>maxUserId):
                blockRowIndex[1] = maxUserId
                
            blockRanges.append((blockRowIndex, blockColIndex))
        stratumIndices[stratum] = blockRanges
        
    for epoch in xrange(T):
        mbc = 0
        for stratum in range(0,num_workers):
            blocks = partitions.filter(lambda x: x[0] == stratum)
            blockRange = stratumIndices[stratum]
            blockInfoList = []
            for index, block in enumerate(blockRange):
                blockRowIndex, blockColIndex = block[0], block[1]
                Wmini = W[blockRowIndex[0]:blockRowIndex[1], :]
                Hmini = H[:, blockColIndex[0]:blockColIndex[1]]
                blockInfo = blocks.filter(lambda x: x[1][0]/block_size[0] == index)
                blockInfoList.append((blockInfo.collect(), Wmini, Hmini, mbc, index, blockRowIndex[0], blockColIndex[0]))
            results = sc.parallelize(blockInfoList, num_workers).map(lambda x: updateGradient(x)).collect()

            #Broadcast W and H and update
            for result in results:           
                blockNum = result[0]
                block = blockRange[blockNum]
                blockRowIndex, blockColIndex = block[0], block[1]
                mbc += result[1]
                W[blockRowIndex[0]:blockRowIndex[1], :] = result[2]
                H[:, blockColIndex[0]:blockColIndex[1]] = result[3]
            
    
    return (W,H) 



    
def main():
    global beta_value
    global lambda_value

    #sc = SparkContext("local", "SGD-Matrix")
    sc = SparkContext(appName="SGD-Matrix")
    #sc = SparkContext("spark://ip-172-31-14-252.us-west-2.compute.internal:7077", "SGD-Matrix")
    num_factors = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_iterations = int(sys.argv[3])
    beta_value = float(sys.argv[4])
    lambda_value = float(sys.argv[5])
    inputV_filepath = sys.argv[6]
    outputW_filepath = sys.argv[7]
    outputH_filepath = sys.argv[8]

    [partitions,W,H,block_size] = load_data(inputV_filepath,num_factors,num_workers,sc)

    W,H = factorize_matrix(partitions,W,H,block_size,num_iterations,num_workers,sc)
    
    # Save W and H
    np.savetxt(outputW_filepath,W,delimiter=',')

    np.savetxt(outputH_filepath,H,delimiter=',')
    
    np.savetxt("predicted_ratings.csv",W.dot(H),delimiter=',')
    
if __name__ == '__main__':
    main()