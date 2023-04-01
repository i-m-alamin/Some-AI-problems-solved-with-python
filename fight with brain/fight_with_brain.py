#Task 01
import numpy as np
file=open('/content/test.txt')

line_1=file.readline().strip()
vertex=int(line_1)+1

line_2=file.readline().strip()
limit=int(line_2)

matrix=np.zeros((vertex,vertex))
for i in range(limit):
  line_3=file.readline().strip()


  Subline_line_3=line_3[0]
  vertex1=int(Subline_line_3)
  
  Subline_line_4=line_3[2]
  vertex_2=int(Subline_line_4)
  
  matrix[vertex1][vertex_2]=1
  matrix[vertex_2][vertex1]=1

Goal=file.readline().strip()
Goal=int(Goal)


arr=[]
for x in range(10):
   arr.append(0)
#print (arr)

q=[]
q.append(0)
 
arr[0]=1
check=[]
for x in range(9):
  check.append(0)


b=0
l=0 
while q:
    
    vis = q[0]
    q.pop(0)
             
         
           
    for i in range(10):
        if (matrix[vis][i] == 1 and
              (arr[i]==0)):
               
               q.append(i)
               arr[i]=1
               check[i]=check[vis]+1+check[i]    
               
               if (matrix[vis][Goal] == 1 and
                        (arr[Goal]==0)):
                         
                       
                       arr[Goal]=1
                       check[Goal]=check[vis]+1+check[Goal]
                       
                       break

print(check[Goal])

#task_2
import numpy as np
#task_2:
def BF_S(matrix,arr,start,Goal):
  q.append(start)
  
  arr[start]=1 
  check=[]
  for x in range(9):
    check.append(0)
  
  while q:
    
    vis = q[0]
    q.pop(0)
            
         
    for i in range(vertex):
        if (matrix[vis][i] == 1 and
              (arr[i]==0)):
              
                  
              
              if ((matrix[vis][Goal]==1) and (arr[Goal]==0)):
                        
                      
                      arr[Goal]=1
                      
                      
                      check[Goal]=check[vis]+1+check[Goal]
                      break
              
              q.append(i)
              arr[i]=1
              check[i]=check[vis]+1+check[i]
                        
  
  return check[Goal]               

file=open('/content/test.txt')

line_1=file.readline().strip()
vertex=int(line_1)

line_2=file.readline().strip()
limit=int(line_2)

matrix=np.zeros((vertex,vertex))
for i in range(limit):
    line_3=file.readline().strip()


    Subline_line_3=line_3[0]
    vertex1=int(Subline_line_3)
    
    Subline_line_4=line_3[2]
    vertex_2=int(Subline_line_4)
    
    matrix[vertex1][vertex_2]=1
    matrix[vertex_2][vertex1]=1

Goal=file.readline().strip()
Goal=int(Goal)


Nora=file.readline().strip()
Nora=int(Nora)


Lara=file.readline().strip()
Lara=int(Lara)


arr=[]
for x in range(10):
    arr.append(0)
#print (arr)

q=[]




b = BF_S(matrix,arr,Nora,Goal)
 


for x in range(0,10):
    arr[x]=0


c = BF_S(matrix,arr,Lara,Goal) 


if (b<c):
  print("Nora")

else :
  print("Lara")

#task_3
import numpy as np

#task_3:
def BF_S(matrix,arr,start,player):
  q.append(start)
  
  arr[start]=1 
  check=[]
  for x in range(vertex):
    check.append(0)
  #print (check)
  while q:
    
    vis = q[0]
    q.pop(0)
            
        
    for i in range(vertex-1,-1,-1):
        if (matrix[vis][i] == 1 and
              (arr[i]==0)):
              
               for x in range(len(player)-1,-1,-1):
                  
                  if (i==player[x]):
                        
                      
                       arr[i]=1
                       
                       q.append(i)
                       check[i]=check[vis]+1+check[i]
                       return check[i]
                       
              
               q.append(i)
               arr[i]=1
               check[i]=check[vis]+1+check[i]
               
               
              
  return check[i]               

file=open('/content/test.txt')

line_1=file.readline().strip()
vertex=int(line_1)

line_2=file.readline().strip()
limit=int(line_2)

matrix=np.zeros((vertex,vertex))
for i in range(limit):
    line_3=file.readline().strip()


    Subline_line_3=line_3[0]
    vertex1=int(Subline_line_3)
    
    Subline_line_4=line_3[2]
    vertex_2=int(Subline_line_4)
    
    matrix[vertex1][vertex_2]=1
    

matrix=matrix.transpose()

Goal=file.readline().strip()
Goal=int(Goal)


participator=file.readline().strip()
participator=int(participator)


player=[]

for x in range(participator):
  state=file.readline().strip()
  state=int(state)
  player.append(state)
  

arr=[]
for x in range(10):
    arr.append(0)
#print (arr)

q=[]

b = BF_S(matrix,arr,Goal,player)
print(b) 
  




