import numpy as np

class Perceptron(object):
  def __init__(self , threshold = 0,alpha=0.01,epoch=100):
    self.alpha = alpha
    self.epoch = epoch
    self.threshold = threshold

  
  def learn(self,inp,out):
    
    
    m = inp.shape[1] #input length
    self.weights = np.zeros(m,float)

    for i in range(0,m):
        self.weights[i] = 1
    self.weights = np.append(self.weights,1)  #adding one threshold input  
    self.errors_ = []
    for i in range(0,self.epoch):

      errors = 0
      #print("input_shape",inp[0].shape,out.shape)

      for x_n,y_n in zip(inp,out):
        x_n = x_n.reshape((m,)) 
        t_n = np.append(x_n,self.threshold) #threshold input appended
        y_i = self.predict(t_n)  
        e = self.alpha*float((y_n[0] - y_i))   #error multiplied by learning rate
        update = np.dot(t_n,e)
        #print(self.weights,y_n,y_i,update,e,t_n)
        self.weights = self.weights + update  #update weights
        
        #print(self.weights)
        errors = errors + np.sum(update != 0.0)/(m)
      self.errors_.append(errors)

    
    #self.threshold = self.weights[2]
    #self.weights = np.delete(self.weights,2)

       
  
  def predict(self, v ):
    #print(self.weights.shape,v.shape) 
    return np.where(np.dot(self.weights,v) >= 0,1,0) #predict to be used while learing
  def calc(self,v):
    v = np.append(v,self.threshold)
    #print(self.errors_)  
    return np.where(np.dot(self.weights,v) >= 0,1,0) #to be used during test
  


'''
OR = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,1]])
OR_Perceptron = Perceptron(1,0.1,20)
OR_Perceptron.learn(OR[:,0:2],OR[:,2:])
print(OR_Perceptron.calc(np.array([0,0])))
'''
AND = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
AND_Perceptron = Perceptron(1,0.1,50)
AND_Perceptron.learn(AND[:,0:2],AND[:,2:])
print(AND_Perceptron.calc([0,0]))
'''
NAND = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,0]])
NAND_Perceptron = Perceptron(1,0.1,50)
NAND_Perceptron.learn(NAND[:,0:2],NAND[:,2:])
print(NAND_Perceptron.calc([1,0]))


NOT = np.array([[0,1],[1,0]])
NOT_Perceptron = Perceptron(1,0.1,15)
NOT_Perceptron.learn(NOT[:,0:1],NOT[:,1:])
print(NOT_Perceptron.calc([1]))'''