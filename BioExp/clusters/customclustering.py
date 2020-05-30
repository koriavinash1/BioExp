import numpy as np
import weakref
from collections import defaultdict
from similarity import computeDistance
from time import time
import math
from pathlib import Path
import pickle
import pprint
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import sys

# Pretty Printer Object for printing with indenting
pp = pprint.PrettyPrinter(indent = 4)

# Class for keeping track of Union Operations and updating the linkage matrix
class _UnionTracker_:
    factor = 1
    def __init__(self, points):
        '''initializer function for UnionTracker'''
        self.points = points
        assert self.points > 1, "Number of points should be greater than 0, {} was provided".format(self.points)
        self.linkage_matrix = np.zeros((self.points-1, 4))

    def union(self, A, B, dist, pts, iteration):
        '''Create a Union Entry in the linkage Matrix'''
        self.linkage_matrix[iteration][0] = A
        self.linkage_matrix[iteration][1] = B
        self.linkage_matrix[iteration][2] = dist*UnionTracker.factor
        self.linkage_matrix[iteration][3] = pts

class _HC_:
    ''' Cluster class for creating and maintaining the clusters for the heirarchical clustering'''
    ClusterCount = 0
    _instances = defaultdict()
    maxClusters = 0

    def __init__(self, key=None, seq=None):
        ''' Initialization of clusters '''
        self._id = Cluster.ClusterCount
        self.initID = self._id
        Cluster.ClusterCount+=1
        Cluster.maxClusters += 1
        self.clusterMembers = dict()
        self._instances[self.initID] = weakref.ref(self)
        self.factor = 1
        if key is not None:
            self.addMember(key, seq)

    def __del__(self):
        ''' Destructor for the Cluster Object'''
        Cluster._instances.pop(self.initID, None)

    def incrementFactor(self, factor=1):
        ''' Increment Multiplication Factor'''
        self.factor += factor

    def addMember(self, key, seq):
        ''' Add memebers (data points) to the cluster, in form of dictionary where the dna sequence is the value'''
        self.clusterMembers[key] = seq

    def destroy(self):
        ''' Explcit destructor call '''
        self.__del__()

    def updateID(self, iteration):
        ''' Update the id of the cluster after merge operation to n+i where i is the iteration number'''
        self._id = Cluster.maxClusters + iteration

    @property
    def memberCount(self):
        '''returns the number of members in the cluster currently '''
        return len(self.clusterMembers.keys())

    @property
    def sequences(self):
        ''' Returns the Member DNA sequences in the cluster '''
        return [self.clusterMembers[key] for key in self.clusterMembers.keys()]

    @classmethod
    def getClusterById(cls, clusterID):
        ''' Fetch cluster object by referencing its id from the Cluster weak reference storage '''
        ref = cls._instances[clusterID]
        obj = ref()
        return obj


    @classmethod
    def generateInitialDistanceMatrix(cls, data=None, test = False):
        ''' Generate the initial nxn distance matrix by computing Distance between each DNA sequence '''

        # For Testing Purpose
        if test == True:
            cls.simMatrix = np.array([[0,9,3,6,11],[9,0,7,5,10],[3,7,0,9,2],[6,5,9,0,8],[11,10,2,8,0]], dtype=float)
        # Actual Dataset Implementation
        else:
            pickleFilePath = Path('data/simMat_3.pkl')
            if pickleFilePath.exists():
                # Load Pickle File storing the simMatrix
                with open(pickleFilePath, 'rb') as file:
                    cls.simMatrix = pickle.load(file)
            else:
                # Compute Distance among DNA Sequence
                cls.simMatrix = np.ones((cls.ClusterCount, cls.ClusterCount))
                for cID in range(cls.ClusterCount):
                    clusterA = cls.getClusterById(cID)
                    for _cID in range(cID, cls.ClusterCount):
                        clusterB = cls.getClusterById(_cID)
                        seq1 = clusterA.sequences[0]
                        seq2 = clusterB.sequences[0]
                        similarity_1 = computeDistance(seq1, seq2)
                        cls.simMatrix[cID, _cID] = similarity_1
                        cls.simMatrix[_cID, cID] = similarity_1
                        print("similarity between {} and {} = {}\r".format(cID, _cID, similarity_1), end='', flush=True)
                        sys.stdout.flush()
                    print('')
                # Save The Pickle File
                with open(pickleFilePath, 'wb') as file:
                    pickle.dump(cls.simMatrix, file)
            # Normalize the Matrix
            # minval = np.amin(cls.simMatrix, axis=(0,1))
            # maxval = np.amax(cls.simMatrix, axis=(0,1))
            # cls.simMatrix = ((cls.simMatrix-minval)/(maxval-minval))

    @classmethod
    def getClusters(cls):
        for key in cls._instances.keys():
            ref = cls._instances[key]
            obj = ref()
            if obj is not None:
                yield obj
            else:
                cls._instances.pop(key, None)


    @classmethod
    def currentClusterCount(cls):
        ''' Returns the currently  existent clusters'''
        return len(cls._instances.keys())


    # Class method to find the minimum distance cluster pair
    @classmethod
    def findMinDistance(cls):
        ''' Find the clusters most similar to each other i.e. with the least distance among them '''
        minDistance = 1*math.inf
        clusterA = None
        clusterB = None
        for rowNumber in range(0,cls.simMatrix.shape[0]-1):
            for colNumber in range(rowNumber+1, cls.simMatrix.shape[1]):
                if cls.simMatrix[rowNumber, colNumber] < minDistance:
                    minDistance = cls.simMatrix[rowNumber, colNumber]
                    clusterA = rowNumber
                    clusterB = colNumber
        return clusterA, clusterB, minDistance

    @classmethod
    def mergeSimilarClusters(cls, mergedRC, toDelete, iteration, dist, heuristic = 'Centroid'):
        ''' Cluster merging and new CLuster Creation based on the preset Heuristic 
            Available Heuristic Values are,
                - Centroid
                - Max
                - Min
        '''
        outdated_m = mergedRC
        outdated_d = toDelete
        
        delCluster = cls.getClusterById(outdated_d)
        toDelete = delCluster._id
        mergedCluster = cls.getClusterById(outdated_m)
        mergedRC = mergedCluster._id

        d_mem = delCluster.memberCount
        m_mem = mergedCluster.memberCount

        if heuristic == 'Centroid': #Compute using Centroid Calculation
            rowSum = (cls.simMatrix[outdated_m, :]*m_mem + cls.simMatrix[outdated_d, :]*d_mem)/(m_mem+d_mem)
        elif heuristic == 'Max':    #Compute using Max Calculation
            rowStack = np.vstack((cls.simMatrix[outdated_m, :], cls.simMatrix[outdated_d, :]))
            rowSum = np.amax(rowStack, axis=0)
        elif heuristic == 'Min':    #Compute using Min Calculation
            rowStack = np.vstack((cls.simMatrix[outdated_m, :], cls.simMatrix[outdated_d, :]))
            rowSum = np.amin(rowStack, axis=0)

        #Update the new row
        cls.simMatrix[:, outdated_m] = rowSum
        cls.simMatrix[outdated_m, :] = rowSum
        cls.simMatrix[:, outdated_d] = 1*math.inf
        cls.simMatrix[outdated_d, :] = 1*math.inf

        #Merge Data Points into the cluster
        for key in delCluster.clusterMembers.keys():
            mergedCluster.addMember(key, delCluster.clusterMembers[key])
        mergedCluster.updateID(iteration)
        mergedCluster.incrementFactor()
        print('Union ({} - {}), distance {}'.format(toDelete, mergedRC, dist))

        # Delete the redundant clusters explicityly
        delCluster.destroy()
        return mergedRC, toDelete, mergedCluster.memberCount, mergedCluster.factor

# Driver Function to execute the Heirarchical Clustering
@timer
def main():
    test = False
    heuristic = 'Centroid'
    reader = DataReader()
    data = reader.loadData()
    dataArray = reader.getDataArray()
    if test == True:
        clusters = [Cluster(dataPoint, data[dataPoint]) for dataPoint in list(data.keys())[:5]]     
    else:
        clusters = [Cluster(dataPoint, data[dataPoint]) for dataPoint in list(data.keys())[:]]      
    Cluster.generateInitialDistanceMatrix(test)
    Uni = UnionTracker(len(clusters))
    print('')
    iteration = 0
    while(Cluster.currentClusterCount() > 1):
        clsA, clsB, dist = Cluster.findMinDistance()
        mergedRC = min(clsA, clsB)
        toDelete = max(clsA, clsB)
        newIDm, newIDd, pts, factor = Cluster.mergeSimilarClusters(mergedRC, toDelete, iteration, dist, heuristic=heuristic)
        Uni.union(newIDd, newIDm, dist, pts, iteration)
        iteration += 1

    labels = list(data.keys())
    drawDendrogram(Uni, labels, heuristic)

def drawDendrogram(UniObj, labels, heuristic):
    ''' Generate the dendrogram using te UniObject's linkage matrix '''
    plt.title("Dendrogram - Agglomerative Clustering -" + heuristic)
    dendrogram(UniObj.linkage_matrix, show_leaf_counts = True, show_contracted = True, labels = labels)
    plt.show()

if __name__ == "__main__":
    main()



class Cluster(object):
    
    def __init__(self, model, weights_pth, layer_name, max_clusters = None, heuristic = 'Centroid'):
        
        """
        model       : keras model architecture (keras.models.Model)
        weights_pth : saved weights path (str)
            metric      : metric to compare prediction with gt, for example dice, CE
            layer_name  : name of the layer which needs to be ablated
            test_img    : test image used for ablation
            max_clusters: maximum number of clusters
        """     

        self.model = model
        self.weights = weights_pth
        self.layer = layer_name
        self.layer_idx = 0
        for idx, layer in enumerate(self.model.layers):
            if layer.name == self.layer:
                self.layer_idx = idx
        self.weights = np.array(self.model.layers[self.layer_idx].get_weights())[0]
        self.heuristic = heuristic
        
    def drawDendrogram(self, UniObj, labels, heuristic):
        ''' Generate the dendrogram using te UniObject's linkage matrix '''
        plt.title("Dendrogram - Agglomerative Clustering -" + heuristic)
        dendrogram(UniObj.linkage_matrix, 
                    show_leaf_counts = True, 
                    show_contracted = True, 
                    labels = labels)
        plt.show()

    def get_clusters(self, threshold=0.5, save_path = None):
        """
        Does clustering on feature space

        save_path  : path to save dendrogram image
        threshold  : fraction of max distance to cluster 
        """

        shape = self.weights.shape
        X = self.weights.reshape(shape[-1], -1)

        position = np.linspace(0, X.shape[-1], X.shape[-1])
        X = X + position[None, :]

        clusters = [_HC_(dataPoint, data[dataPoint]) for dataPoint in data]      
        Cluster.generateInitialDistanceMatrix(test)
        Uni = _UnionTracker_(len(clusters))

        iteration = 0
        while(Cluster.currentClusterCount() > 1):
            clsA, clsB, dist = Cluster.findMinDistance()
            mergedRC = min(clsA, clsB)
            toDelete = max(clsA, clsB)
            newIDm, newIDd, pts, factor = Cluster.mergeSimilarClusters(mergedRC, 
                                                                    toDelete, 
                                                                    iteration, 
                                                                    dist, 
                                                                    heuristic=self.heuristic)
            Uni.union(newIDd, newIDm, dist, pts, iteration)
            iteration += 1

        self.drawDendrogram(Uni, labels, heuristic)
        
        plt.figure(figsize=(20, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        R = dendrogram(linkage_matrix, truncate_mode='level')
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, '{}_dendrogram.png'.format(self.layer)), bbox_inches='tight')
        else:
            plt.show()

        return labels 


    def plot_weights(self, labels, save_path=None):
        """
        dim x: k x k x in_c x out_c
        """
        shape = self.weights.shape
        
        features = []
        for label in np.unique(labels):
            wts_idx = np.where(labels==label)[0]
            wts = self.weights[:,:,:,wts_idx].T
            wts = wts.reshape(len(wts_idx), -1)
            
            features.extend(wts)
            features.extend(np.zeros((3, wts.shape[1])))
            """
            feature = np.zeros((s, shape[1]*cls))
            for ii in wt_idx:
                wt = self.weights[:,:,:, ii]
                for i in range(rws):
                    for j in range(cls):
                        try:
                            feature[i*shape[0]: (i + 1)*shape[0], 
                                j*shape[1]: (j + 1)*shape[1]] = wt[:, :, j*rws + i]
                        except:
                            pass

                plt.clf()
                plt.imshow(feature)
                if not save_path:
                    plt.show()
                else:
                    os.makedirs(save_path, exist_ok = True)
                    plt.savefig(os.path.join(save_path, 'cluster_{}_idx_{}.png'.format(label, ii)), bbox_inches='tight')
            """
            plt.clf()
            plt.imshow(wts)
            if not save_path:
                plt.show()
            else:
                os.makedirs(save_path, exist_ok = True)
                plt.savefig(os.path.join(save_path, 'layer_{}__concept_{}.png'.format(self.layer_idx, label)), dpi=2000, bbox_inches='tight')
        
        plt.clf()
        plt.imshow(features)
        if not save_path:
            plt.show()
        else:
            os.makedirs(save_path, exist_ok = True)
            plt.savefig(os.path.join(save_path, 'layer_{}__all_concepts.png'.format(self.layer_idx)), dpi=2000, bbox_inches='tight')