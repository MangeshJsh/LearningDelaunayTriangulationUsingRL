
# coding: utf-8

# In[1]:


class Point():
    def __init__(self, pid, x, y):
        self.x = x
        self.y = y
        self.pid = pid


# In[2]:


class Edge():
    def __init__(self, eid, pid1, pid2):
        self.eid = eid
        self.pid1 = pid1
        self.pid2 = pid2

