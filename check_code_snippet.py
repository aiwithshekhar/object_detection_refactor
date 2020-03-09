import numpy as np

# s = [np.zeros(())]
# # s=[[]*10]
# val=np.array([10,20,30,40,50,10,1,])
#
# for i in range (len(s)):
#     s[i]=val
#     # val+=100
l=np.array([[1,2,3,4,5],[11,21,31,41,51],[21,22,23,24,25]])
l1=np.array([[1,2,3,4,5],[11,21,31,41,51],[21,22,23,24,25]])
# l2=np.ones((0,5))


d = [np.zeros((0,5))]*3

d[0]=l
# d[1]=l2
d[2]=l1


# for i in d:
#     print (type(i))
# labels = np.concatenate(d, 0)
#
for i in d:
    print (i.shape)