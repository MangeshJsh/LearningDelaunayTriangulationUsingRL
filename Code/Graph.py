
# coding: utf-8

# In[7]:


import networkx as nx
from CommonDefs import Point, Edge


# In[8]:


class Graph():
    def __init__(self, points):
        self.g = nx.Graph()
        nodeIds = []
        coords = {}
        for point in points:
            nodeIds.append(point.pid)
            coords[point.pid] = [point.x, point.y]
        self.g.add_nodes_from(nodeIds)
        nx.set_node_attributes(self.g, coords, 'pos')   
    
    def getNumberOfPoints(self):
        return self.g.number_of_nodes()
    
    def hasEdge(self, pid1, pid2):
        return self.g.has_edge(pid1, pid2)
    
    def addEdge(self, pid1, pid2):
        self.g.add_edge(pid1, pid2)
    
    def getNodeIdFromPosAttr(self, x, y):
        for key, value in self.g.nodes.data('pos'):
            if x==value[0] and y==value[1]:
                return key
        return -1
    
    def getCoordsFromNodeId(self, nodeId):
        return self.g.nodes[nodeId]['pos']
            
    def drawGraph(self):
        nx.draw(self.g, pos=nx.get_node_attributes(self.g, 'pos'), with_labels=True)
    
    def clearGraph(self):
        self.g.clear()
            
    def getAdjacencyMatrix(self):
        return nx.adjacency_matrix(self.g).todense()
    
    def getTriangles(self):
        numNodes = nx.number_of_nodes(self.g)
        adjMtx = nx.adjacency_matrix(self.g).todense()
        triangles = set()
        for i in range(0, numNodes):
            for j in range(0, numNodes):
                for k in range(0, numNodes):
                    # check the triplet if it satisfies the condition
                    if(i != j and i != k and j != k and adjMtx[i,j] and adjMtx[j,k] and adjMtx[k,i]):
                        triIndices = (i+1, j+1, k+1)
                        triIndices = sorted(triIndices)
                        triangles.add(tuple(triIndices))
        return triangles    

