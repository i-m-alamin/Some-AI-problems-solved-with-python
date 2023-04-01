def fitness(population, n):

  m=int (n*((n-1)/2))   
  vertical=0
  diagonal=0
   
  for i in range(n):
    for j in range(i+1,n):
      if population[i] == population[j]:
        vertical+=1
      elif abs(i-j) == abs(population[i]-population[j]):
        diagonal+=1
  tolal= vertical+diagonal
         
  to=m-tolal
  
  return to


def select(fit):
  
  c=np.random.randint(0,len(population))
  

  
  return population[c]

def crossover(x, y):
  
   
  z=np.random.randint(0,len(x) ,len(x))
  
  c=np.random.randint(0,len(x))
  
  z[:c]=x[:c]
  
  z[c:]=y[c:]
  
     
  return z


def mutate(child):
  
  c=np.random.randint(0,len(child))
  z= np.random.randint(0,len(child))
  child[c]=z   
 
  

  return child 



def GA (population, n, mutation_threshold):
  
  max_fit=fitness(population[0],n)
  fit=max_fit
  inde=population[0]
  
  
  
  for k in range(11):
    pro=[]
    
 
    for i in range(10):
      x=select(fit)  
      y=select(fit)
      
      child=crossover(x,y)
      
      if np.random.uniform(0,1) < mutation_threshold:
        child=mutate(child) 
      
      fit=fitness(child,len(child))
      if (fit>max_fit):
        max_fit=fit
        inde=child
        
      pro.append(child)
     
    pro.append(population[np.random.randint(0,len(population))])  
    #for i in range(len(pro)):
      #print(pro[i])
    #print()     
  return max_fit,inde



'''main part'''
'''for 8 queen problem, n = 8'''
import numpy as np
n = 8

'''start_population denotes how many individuals/chromosomes are there
  in the initial population n = 8'''
start_population = 10 

'''if you want you can set mutation_threshold to a higher value,
   to increase the chances of mutation'''
mutation_threshold = 0.3

'''creating the population with random integers between 0 to 7 inclusive
   for n = 8 queen problem'''
population = np.random.randint(0, n, (start_population, n))
#print(population)


value,index = GA(population, n, mutation_threshold)
print("Maximum fitness: ",value)
print("Board : ",index)

