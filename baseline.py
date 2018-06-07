
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable

import cv2
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')


# In[2]:


class StarNet(nn.Module):

    def __init__(self):
        super(StarNet, self).__init__()
        
        self.features = nn.Sequential(
            
            # conv1
            nn.Conv2d(3, 56, kernel_size=5),
            nn.ReLU(inplace=True),
            # conv2
            nn.Conv2d(56, 12, kernel_size=1),
            nn.ReLU(inplace=True),
            # conv22
            nn.Conv2d(12, 12, kernel_size=3),
            nn.ReLU(inplace=True),
            # conv23
            nn.Conv2d(12, 12, kernel_size=3),
            nn.ReLU(inplace=True),
            # conv24
            nn.Conv2d(12, 12, kernel_size=3),
            nn.ReLU(inplace=True),
            # conv25
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # conv26
            nn.Conv2d(12, 56, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            # conv3
            nn.Conv2d(56, 1, kernel_size=9, padding=4, stride=4),
            
        )


    def forward(self, x):
        x = self.features(x)
        return x


# In[3]:


starnet = StarNet()


# In[4]:


starnet


# In[5]:


image = cv2.imread('/home/guang/SuperRes/images_1080p/starcraft_1080p_0_image_001.bmp')
image_input = image[400:656,600:856, :];


# In[6]:


plt.imshow(image_input)


# In[7]:


t = torch.from_numpy(image_input).contiguous()
t.size()


# In[8]:


t = t.view([1] + list(t.size()))
t.size()


# In[9]:


x = Variable(t)


# In[ ]:


starnet(x)


# In[ ]:





# In[ ]:





# In[ ]:




