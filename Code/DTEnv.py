# -*- coding: utf-8 -*-

from CommonDefs import Point
from TwoDimConvexHull import TwoDimConvexHull, PrintTwoDimConvexHull
from Utils import nearestKNeighboursOfEdgeMidPt, checkTriangleForDelaunayCriteria
from Graph import Graph

import numpy as np
import math

class DTEnv():
    def __init__(self, k=-1, filterPointsFormingTriangleOnEdge = False):
        self.numNearestNeighbors = k
        self.filterPointsFormingTriangleOnEdge = filterPointsFormingTriangleOnEdge
        self.edgesToProcess = []
        self.generatedTriangles = []
        self.points = []
        self.hull = []
        self.graph = None
        
    def initialize(self, points):
        self.points.extend(points)
        self.graph = Graph(points)
        
    def reset(self):
        self.edgesToProcess.clear()
        self.generatedTriangles.clear()
        self.points.clear()
        if self.graph is not None:
            self.graph.clearGraph()
        
    def getStartState(self):
        convexHull = TwoDimConvexHull(self.points)
        hull = convexHull.getConvexHull() #Point objects representing vertices of the convex hull
        
        edges = []
        
        for i in range(len(hull) - 1):
            edges.append([hull[i].pid, hull[i+1].pid])
            
        edges.append([hull[len(hull) - 1].pid, hull[0].pid])
        
        for i in range(len(edges)):
            edges[i] = sorted(edges[i])
        
        self.hull = edges
        
        randomHullEdge = np.random.randint(0, len(edges))
        self.graph.addEdge(edges[randomHullEdge][0], edges[randomHullEdge][1]) #select random edge from the convex hull for starting state
        self.edgesToProcess.append([edges[randomHullEdge][0], edges[randomHullEdge][1]])
        return [edges[randomHullEdge][0], edges[randomHullEdge][1]], np.array(self.graph.getAdjacencyMatrix()).flatten()
        
    # edge is the list of start point and end point ids forming the edge
    def removeProcessedEdge(self, edge):
        self.edgesToProcess.remove(edge)
    
    def getNodeIdFromPosAttr(self, x, y):
        return self.graph.getNodeIdFromPosAttr(x,y)
    
    '''
    Each point selection is treated as an action. The point is encoded wrt to the edge
    for which we need to select the new point to form the delaunay triangle. Point is encoded
    as following vector:
    [pt.x, pt.y, pt to pid1 dist, pt to pid2 dist, angle1 with edge, angle 2 with edge ] 
    '''
    def generatePointsEncodingWrtEdge(self, pid1, pid2, points):
        pointsEncodings = []
        edgeStPt = self.getPointFromId(pid1)
        edgeEnPt = self.getPointFromId(pid2)
        for ptId in points:
            pt = self.getPointFromId(ptId)
            ptEncoding = [edgeStPt.x, edgeStPt.y, edgeEnPt.x, edgeEnPt.y, pt.x, pt.y]
            aSquared = ((pt.x - edgeStPt.x)**2) + ((pt.y - edgeStPt.y)**2)
            bSquared = ((pt.x - edgeEnPt.x)**2) + ((pt.y - edgeEnPt.y)**2)
            cSquared = ((edgeStPt.x - edgeEnPt.x)**2) + ((edgeStPt.y - edgeEnPt.y)**2)
            a = math.sqrt(aSquared)
            b = math.sqrt(bSquared)
            c = math.sqrt(cSquared)
            angle1 = math.acos((bSquared + cSquared - aSquared)/(2*b*c))
            angle2 = math.acos((aSquared + bSquared - cSquared)/(2*a*b))
            angle3 = math.acos((aSquared + cSquared - bSquared)/(2*a*c))
            semiP = (a + b + c)/2
            area = math.sqrt(semiP * (semiP-a) * (semiP-b) * (semiP-c))
            area = (1 / area) * 0.01
            ptEncoding.extend([a,b,c,angle1, angle2, angle3, area])
            pointsEncodings.append(ptEncoding)
        return pointsEncodings
    
    def getNumberOfPoints(self):
        return self.graph.getNumberOfPoints()
    
    def getPointFromId(self, pid):
        for point in self.points:
            if (point.pid == pid):
                return point
            
    def getStateActionEncoding(self, state, action):
        stateActionEncoding = []
        stateActionEncoding.extend(state)
        stateActionEncoding.extend(action)
        #stateActionEncoding = np.reshape(stateActionEncoding, [1, len(stateActionEncoding)])
        return stateActionEncoding
        
    def getCurrentStateAdjMat(self):
        convexHull = TwoDimConvexHull(self.points)
        hull = convexHull.getConvexHull() #Point objects representing vertices of the convex hull
        edges = []
        for i in range(len(hull) - 1):
            edges.append([hull[i].pid, hull[i+1].pid])
        edges.append([hull[len(hull) - 1].pid, hull[0].pid])
        randomHullEdge = np.random.randint(0, len(edges))
        self.graph.addEdge(edges[randomHullEdge][0], edges[randomHullEdge][1]) #select random edge from the convex hull for starting state
        self.edgesToProcess.append([edges[randomHullEdge][0], edges[randomHullEdge][1]])    
        return [edges[randomHullEdge][0], edges[randomHullEdge][1]], self.graph.getAdjacencyMatrix()
    
    
    def filterValidActions(self, edge, sortedNearestPts):
        adjMat = self.graph.getAdjacencyMatrix()
        pt1Row = adjMat[edge[0]-1][0]
        pt2Row = adjMat[edge[1]-1][0]
        res = pt1Row + pt2Row
        res = np.array(res).flatten()
        count = len([x for x in res if x == 2])
        if count == 1: 
            #there should be only one triangle on the edge
            for i in range(len(res)):
                if res[i] > 1:
                    thirdTriPt = i + 1
            ept1 = self.graph.getCoordsFromNodeId(edge[0])
            ept2 = self.graph.getCoordsFromNodeId(edge[1])
            tpt = self.graph.getCoordsFromNodeId(thirdTriPt)
            vec1 = [ept2[0] - ept1[0], ept2[1] - ept1[1]]
            vec2 = [tpt[0] - ept1[0], tpt[1] - ept1[1]]
            
            triSign = np.sign(np.cross(vec1, vec2))
            pointsToRemove = []
            for i in range(len(sortedNearestPts)):
                testPtCoords = self.graph.getCoordsFromNodeId(sortedNearestPts[i])
                testVec = [testPtCoords[0] - ept1[0], testPtCoords[1] - ept1[1]]
                if triSign == np.sign(np.cross(vec1, testVec)):
                    pointsToRemove.append(sortedNearestPts[i])
            for i in range(len(pointsToRemove)):
                sortedNearestPts.remove(pointsToRemove[i])
            return sortedNearestPts
        return sortedNearestPts
    
    '''
    This function takes an edge: list containing start point and end point indices,
    computes k nearest neighbors to the edge midpoint excluding edge end points,
    returns point ids and corresponding adj matrix containing new edges information as the possible actions
    The state adj matrix and this action matrix when added will give the new state indicating new triangle formation
    '''
    def getPossibleActions(self, edge):
        pt1 = self.getPointFromId(edge[0])
        pt2 = self.getPointFromId(edge[1])
        midPoint = Point(-1, (pt1.x + pt2.x)/2, (pt1.y + pt2.y)/2)
        sortedNearestPts = nearestKNeighboursOfEdgeMidPt(edge[0], edge[1], midPoint, self.points, -1)

        if not sortedNearestPts:
            return []
        
        filteredPts = self.filterValidActions(edge, sortedNearestPts)
        
        kNearestValidActions = []
        if self.numNearestNeighbors != -1:
            if len(filteredPts) <= self.numNearestNeighbors:
                kNearestValidActions.extend(filteredPts)
            elif len(filteredPts) > self.numNearestNeighbors:
                for i in range(0, self.numNearestNeighbors):
                    kNearestValidActions.append(filteredPts[i])
        else:
            kNearestValidActions.extend(filteredPts)
            
        if not filteredPts:
            kNearestValidActions.extend(sortedNearestPts)
        
        possibleActionsEncodings = self.generatePointsEncodingWrtEdge(edge[0], edge[1], kNearestValidActions)
        return possibleActionsEncodings
    
    def getPossibleActionsForNewState(self, oldState, newState):
        oldStateAdjMat = np.reshape(oldState, [self.getNumberOfPoints(), self.getNumberOfPoints()])
        newStateAdjMat = np.reshape(newState, [self.getNumberOfPoints(), self.getNumberOfPoints()])
        newEdgesMat = newStateAdjMat - oldStateAdjMat
        newedges = []
        for i in range(newEdgesMat.shape[0]):
            for j in range(newEdgesMat.shape[1]):
                if newEdgesMat[i][j] == 1:
                    newedges.append([i+1,j+1])
        allPossibleActionsForState = []
        for i in range(len(newedges)):
            allPossibleActionsForState.extend(self.getPossibleActions(newedges[i]))
        return allPossibleActionsForState
    
    def getNextState(self, edge, action):
        chosenPtId = self.graph.getNodeIdFromPosAttr(action[4], action[5])       
             
        edge1 = sorted([edge[0], chosenPtId])
        
        if ((edge1 not in self.edgesToProcess) and (self.graph.hasEdge(edge1[0], edge1[1]) == False) and (edge1 not in self.hull)): 
            self.edgesToProcess.append(edge1)
            
        edge2 = sorted([edge[1], chosenPtId])
        
        if ((edge2 not in self.edgesToProcess) and (self.graph.hasEdge(edge2[0], edge2[1]) == False) and (edge2 not in self.hull)):
            self.edgesToProcess.append(sorted(edge2))
        
        self.graph.addEdge(edge[0], chosenPtId)
        self.graph.addEdge(edge[1], chosenPtId)
        #print('new state:')
        #self.graph.drawGraph()
        return np.array(self.graph.getAdjacencyMatrix()).flatten()
    
    # Each edge should be processed only once since each new edge already has associated triangle
    # and each manifold edge should have only one or two triangles on the edge. One triangle in case of 
    # boundary edge. Returns the list of edges to process. Each element is the list containing the start
    # and end points of the edge
    def getEdgesToProcess(self):
        return self.edgesToProcess
    
    def getReward(self, edge, action):
        P = self.getPointFromId(edge[0])
        Q = self.getPointFromId(edge[1])
        chosenPtId = self.graph.getNodeIdFromPosAttr(action[4], action[5])
        R = self.getPointFromId(chosenPtId)
        chosenTriangle = [edge[0], edge[1], chosenPtId]
        chosenTriangle = sorted(chosenTriangle)
        if chosenTriangle in self.generatedTriangles:
            #print('Triangle already present')
            return -0.1
        self.generatedTriangles.append(chosenTriangle)
        if (checkTriangleForDelaunayCriteria(P, Q, R, self.points) == True):
            return 1
        return -1
    
    def getGeneratedTriangles(self):
        return self.generatedTriangles
    
    def isTerminalState(self):
        if not self.edgesToProcess:
            return True
        return False
    
    def drawGraph(self):
        self.graph.drawGraph()

