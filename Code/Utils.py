
# coding: utf-8

# In[29]:


import math
from CommonDefs import Point, Edge


# In[30]:


'''
Function takes the start and end point ids representing the edge, mid point of the edge,
the input points and number of nearest points required. It returns the points sorted in ascending 
order as per the distance from the edge mid point. It excludes the edge end points from the nearest points.
'''
def nearestKNeighboursOfEdgeMidPt(startPtId, endPtId, midPointOfEdge, targetPoints, k=-1):
    nearestPoints = {}
    for point in targetPoints:
        nearestPoints[point.pid] = ((midPointOfEdge.x - point.x)**2) + ((midPointOfEdge.y - point.y)**2)
    
    if startPtId in nearestPoints:
        nearestPoints.pop(startPtId)
    
    if endPtId in nearestPoints:
        nearestPoints.pop(endPtId)
    
    sortedPts = sorted(nearestPoints.items(), key=lambda nearestPoints: nearestPoints[1])  
    
    sortedPtIds = []
    
    if (k != -1):
        for i in range(0, k):
            sortedPtIds.append(sortedPts[i][0])
    else:
        for i in range(len(sortedPts)):
            sortedPtIds.append(sortedPts[i][0])
    
    return sortedPtIds


# In[31]:


# Function to find the line given two points
def lineFromPoints(P, Q, a, b, c):
    a = Q.y - P.y
    b = P.x - Q.x
    c = a * (P.x) + b * (P.y)
    return a, b, c


# In[32]:


# Function which converts the input line to its
# perpendicular bisector. It also inputs the points
# whose mid-point lies on the bisector
def perpendicularBisectorFromLine(P, Q, a, b, c):
    mid_point = [(P.x + Q.x)/2, (P.y + Q.y)/2]
    #print('midpoint X, Y: {} , {}'.format(mid_point[0], mid_point[1])) 
    # c = -bx + ay
    c = -b * (mid_point[0]) + a * (mid_point[1])
    temp = a
    a = -b
    b = temp
    return a, b, c


# In[33]:


# Returns the intersection point of two lines
def lineLineIntersection(a1, b1, c1, a2, b2, c2):
    determinant = a1 * b2 - a2 * b1
    if (determinant == 0):
        print('Determinant is zero')
        print([a1, b1, c1, a2, b2, c2])
           
        # The lines are parallel. This is simplified
        # by returning a pair of (10.0)**19
        return [(10.0)**19, (10.0)**19]
    else:
        x = (b2 * c1 - b1 * c2)/determinant
        y = (a1 * c2 - a2 * c1)/determinant
        return [x, y]


# In[34]:


def findCircumCenter(P, Q, R):
   
    # Line PQ is represented as ax + by = c
    a, b, c = 0.0, 0.0, 0.0
    a, b, c = lineFromPoints(P, Q, a, b, c)
 
    # Line QR is represented as ex + fy = g
    e, f, g = 0.0, 0.0, 0.0
    e, f, g = lineFromPoints(Q, R, e, f, g)
 
    # Converting lines PQ and QR to perpendicular
    # vbisectors. After this, L = ax + by = c
    # M = ex + fy = g
    a, b, c = perpendicularBisectorFromLine(P, Q, a, b, c)
    e, f, g = perpendicularBisectorFromLine(Q, R, e, f, g)
 
    # The point of intersection of L and M gives
    # the circumcenter
    circumcenter = lineLineIntersection(a, b, c, e, f, g)
 
    #if (circumcenter[0] == (10.0)**19 and circumcenter[1] == (10.0)**19):
        #print("The two perpendicular bisectors found come parallel")
        #print("Thus, the given points do not form a triangle and are collinear")
    #else:
        #print("The circumcenter of the triangle PQR is: ", end="")
        #print("(", circumcenter[0], ",", circumcenter[1], ")")
    
    return circumcenter


# In[35]:


def pointIsInsideCircumcircle(circumCenter, triangleVertex, pointToTest):
    # Radius of the circumcenter of the triangle
    radius = ((circumCenter[0] - triangleVertex[0]) ** 2) + ((circumCenter[1] - triangleVertex[1]) ** 2)
 
    # Distance between point and circumcenter 
    dis = ((circumCenter[0] - pointToTest[0]) ** 2) + ((circumCenter[1] - pointToTest[1]) ** 2)
    if (dis < radius):
        return True
    return False


# In[36]:


def checkTriangleForDelaunayCriteria(P, Q, R, targetPoints):
    circumcenter = findCircumCenter(P, Q, R)
    #print(circumcenter[0])
    #print(circumcenter[1])
    for pt in targetPoints:
        if pointIsInsideCircumcircle(circumcenter, [P.x, P.y], [pt.x, pt.y]) == True:
            return False
    return True

