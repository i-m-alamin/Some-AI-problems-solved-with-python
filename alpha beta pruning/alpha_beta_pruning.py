#max-min Algorithm

import math
i_M=0
def minimax (position,depth,maximizingplayer,le,myval):
  global i_M
  if depth == 0:
    i_M+=1
    return le[myval]
  
  if maximizingplayer:
    maxival= -math.inf
    for x in range((position*3)+1,(position*3)+4):
      
      if x<len(le):
        eval=minimax (x,depth-1,False,le,x)
        maxival=max(maxival,eval)
        myval=maxival
        #print(" MAX ",maxival)
    return myval,i_M
  else :
    minival= +math.inf
    for x in range((position*3)+1,(position*3)+4):
       if x<len(le):
         eval=minimax (x,depth-1,True,le,x)
         minival=min(minival,eval)
         myval=minival
         #print(" MIN ",minival)
    return myval


# Alpha Beta Pruning  Algorithm 
import math
i_A=0
def max_value (position,depth,le,a,B):
  global i_A
  if depth == 0:
    i_A+=1
    return le[position]

  v=-math.inf
  for x in range((position*3)+1,(position*3)+4):
    eval=min_value (x,depth-1,le,a,B)
    v=max(v,eval)
    if v >= B:
      return v  
    a=max(a,v)
    #print(" MAX ",a)
  return a,i_A


def min_value (position,depth,le,a,B):
  global i_A
  if depth == 0:
    i_A+=1
    return le[position]
    
    
  v= math.inf
  for x in range((position*3)+1,(position*3)+4):
      
    eval=max_value (x,depth-1,le,a,B)
    v=min(v,eval)
    if v <= a:
      return v  
    B=min(B,v)
    
    #print(" MIN ",B)
  return v

def Alpha_Beta_Search(State):
  v,i_A= max_value(0,2,State,-math.inf,math.inf)
  return v,i_A 



import numpy as np
import random

file=open('/content/test03.txt')
line_1=file.readline().strip()
turn=int(line_1)
line_2=file.readline().strip()
branch=int(line_2)
line_3=file.readline().strip()
sub_line_3=line_3[0]
min_node=int(sub_line_3)
sub_line_3=line_3[1:]
max_node=int(sub_line_3)

depth=2*turn
print("Depth:",depth)
print("Branch:",branch)
Terminal_States=pow(3,depth)
print("Terminal States (Leaf Nodes):",Terminal_States)
length=0
for i in range(branch):
  length=length+pow(3,i)
  
pos=[]
for i in range(length):
  pos.append(random.randint(min_node,max_node))
#print(pos)  

M_number,i_M= minimax (0,2,True,pos,4200)

A_number,i_A= Alpha_Beta_Search(pos)
if M_number == A_number:
  print("Maximum amount:",A_number)
print("Comparisons:",i_M)
print("Comparisons:",i_A)
 

