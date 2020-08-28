#%matplotlib inline
#import dgl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import csv 
import io
import os
import sys
import random
#import torch
from copy import copy, deepcopy
from random import uniform, choice, randint
from pylab import *
from operator import itemgetter
#from google.colab import drive
#from google.colab import files

#!ls

def readfile(xyzfile):
  filename=xyzfile
  xyz = open(filename, "r")
  xyz.readline()
  xyz.readline()
  X=[]
  Y=[]
  Z=[]
  for line in xyz:
    line.split()
    atom, x, y, z = line.split()
    X.append(float(x))
    Y.append(float(y))
    Z.append(float(z))
  xyz.close()
  a=np.array(list(zip(X, Y, Z)))
  return a

cluster=readfile("CO_1.xyz")
cluster

ECR = {'hidrogeno':31, 'H':31, '1':31,'helio':28, 'He':28, '2':28, 
       "litio":128, 'Li':128, '3':128,'berilio':96, 'Be':96, '4':96,
       'boro':84, 'B':84, '5':84,'carbono':73, 'C':73, '6':73,
       'nitrogeno':71, 'N':71, '7':71,'oxigeno':66, 'O':66, '8':66,
       'fluor':57 , 'F':57 , '9':57 ,'neon':58 , 'Ne':58 , '10':58 ,
       'sodio':166 , 'Na':166, '11':166 ,'magnesio':141 , 'Mg':141 , '12':141 ,
       'aluminio':121 , 'Al':121 , '13':121 ,'silicio':111 , 'Si':111 , '14':111 ,
       'fosforo':107 , 'P':107 , '15':107 , 'azufre':105 , 'S':105 , '16':105 ,
       'cloro':102 , 'Cl':102 , '17':102 ,'argon':106 , 'Ar':106 , '18':106 ,
       'potasio':203 , 'K':203 , '19':203 , 'calcio':176 , 'Ca':176 , '20':176 ,
       'escandio':170 , 'Sc':170 , '21':170 , 'titanio':160 , 'Ti':160 , '22':160 ,
       'vanadio':153 , 'V':153 , '23':153 , 'cromo':139 , 'Cr':139 , '24':139 , 
       'manganeso':139 , 'Mn':139 , '25':139 ,'hierro':132 , 'Fe':132 , '26':132 ,
       'cobalto':126 , 'Co':126 , '27':126 ,'niquel':124 , 'Ni':124 , '28':124 ,
       'cobre':132 , 'Cu':132 , '29':132 ,'cinc':122 , 'Zn':122 , '30':122 , 
       'galio':122 , 'Ga':122 , '31':122 ,'germanio':120 , 'Ge':120 , '32':120 ,
       'arsenico':119 , 'As':119 , '33':119 ,  'selenio':120 , 'Se':120 , '34':120 , 
       'bromo':120 , 'Br':120 , '35':120 , 'kripton':116 , 'Kr':116 , '36':116 , 
       'rubidio':220 , 'Rb':220 , '37':220 , 'estroncio':195 , 'Sr':195 , '38':195 ,
       'ytrio':190 , 'Y':190 , '39':190 , 'circonio':175 , 'Zr':175 , '40':175 , 
       'niobio':164 , 'Nb':164 , '41':164 , 'molibdeno':154 , 'Mo':154 , '42':154 ,
       'tecnecio':147 , 'Tc':147 , '43':147 , 'rutenio':146 , 'Ru':146 , '44':146 ,
       'rodio':142 , 'Rh':142 , '45':142 , 'paladio':139 , 'Pd':139 , '46':139 ,
       'plata':145 , 'Ag':145 , '47':145 , 'cadmio':144 , 'Cd':144 , '48':144 ,
       'indio':142 , 'In':142 , '49':142 ,  'estano':139 , 'Sn':139 , '50':139 ,
       'antimonio':139 , 'Sb':139 , '51':139 , 'teluro':138 , 'Te':138 , '52':138 , 
       'yodo':139 , 'I':139 , '53':139 , 'xenon':140 , 'Xe':140 , '54':140 ,
       'cesio':244 , 'Cs':244 , '55':244 ,  'bario':215 , 'Ba':215 , '56':215 ,
       'hafnio':187 , 'Hf':187 , '72':187 , 'tantalo':170 , 'Ta':170 , '73':170 ,
       'wolframio':162 , 'W':162 , '74':162 , 'renio':151 , 'Re':151 , '75':151 ,
       'osmio':144 , 'Os':144 , '76':144 , 'iridio':141 , 'Ir':141 , '77':141 ,
       'platino':136 , 'Pt':136 , '78':136 ,  'oro':136 , 'Au':136 , '79':136 ,
       'mercurio':132 , 'Hg':132 , '80':132 , 'talio':145 , 'Tl':145 , '81':145 , 
       'plomo':146 , 'Pb':146 , '82':146 , 'bismuto':148 , 'Bi':148 , '83':148 ,
       'polonio':140 , 'Po':140 , '84':140 , 'astato':150 , 'At':150 , '85':150 ,
       'radon':150 , 'Rn':150 , '86':150 ,'francio':260 , 'Fr':260 , '87':260 ,
       'radio':221 , 'Ra':221, '88':221}
(ECR['Ag']*2)/100

def aristas(elemento, cluster):
  aristas=[]
  for i in range(0, len(cluster)):
    for j in range(i+1, len(cluster)):
      dist=( ((cluster[i][0]-cluster[j][0])**2) + ((cluster[i][1]-cluster[j][1])**2) + ((cluster[i][2]-cluster[j][2])**2))**0.5
      if dist < (ECR[elemento]*2)/100:
        aristas.append((i, j))
  return aristas

def distancia(elemento, cluster):
  distance=[]
  for i in range(0, len(cluster)):
    for j in range(i+1, len(cluster)):
      dist=( ((cluster[i][0]-cluster[j][0])**2) + ((cluster[i][1]-cluster[j][1])**2) + ((cluster[i][2]-cluster[j][2])**2))**0.5
      #print("la distancia de", i+1 ,"a", j+1 ,"es")
      #print(c)
      if dist < (ECR[elemento]*2)/100:
        distance.append(dist)
        #print("La distancia de", i+1 ,"a", j+1 ,"es")
        #print(c)
  return distance

class Vecinos():
  pass

class uss():
  pass
class atom():
  numero=''
  x=''
  y=''
  z=''
  vecinos=Vecinos()
  u=uss()
  vi=[]
  #vj=[]
  z=[]

#a partir de aqui voya intentar una serie de ideas que se me estuvieron ocurriendo este fin de semana a ver que tan bien o mal salen
kuroster=[]
for i in range(0, len(cluster)):
  kuroster.append(atom())

aver=np.c_[np.zeros(len(cluster)),cluster]
for i in range(0,len(cluster)):  
  kuroster[i].vecinos=np.c_[aver,np.zeros(len(cluster))]
  kuroster[i].numero=i
  kuroster[i].x=cluster[i][0]
  kuroster[i].y=cluster[i][1]
  kuroster[i].z=cluster[i][2]



for i in range(0,len(cluster)):
  for j in range(0, len(cluster)):
    kuroster[i].vecinos[j][0]=j
    kuroster[i].vecinos[j][4]=( ((cluster[i][0]-cluster[j][0])**2) + ((cluster[i][1]-cluster[j][1])**2) + ((cluster[i][2]-cluster[j][2])**2))**0.5

#no entiendo porquer esto no esta borrando la fila del atomo 
for i in range(0,len(cluster)):
  np.delete(kuroster[i].vecinos,i,0)

for i in range(0, len(cluster)):
  kuroster[i].u=np.zeros((len(cluster),41))


for i in range(0,len(cluster)):
  for j in range(0,len(cluster)):
    for k in range(0, 41):
      kuroster[i].u[j][k]=exp(-((kuroster[i].vecinos[j][4]-(0.2*k))**2)/(0.2)**2)

r=[]

for i in range(len(cluster)):
  r=np.random.rand(1,64)
  kuroster[i].vi=[]
  kuroster[i].vi=r
 # print(r)
  #kuroster[i].vi.append(r)
  #print(i)

#print('sali del for')
#kuroster[0].vi
#generar W y b
w1=np.random.randn(1,64)
w2=np.random.randn(1,64)
b1=np.random.randn(64)
b2=np.random.randn(64)



#aqui hay que meter las vueltas para iterar
for i in range(0,len(cluster)):
    kuroster[i].z=[]
    for j in range(0,len(cluster)):
        temp=np.hstack((kuroster[i].vi[0],kuroster[j].vi[0]))
        kuroster[i].z.append(np.hstack((temp,kuroster[i].u[j])))
        
#for (item1, item2) in zip(w1, b1):
    #sum_list.append(item1+item2)
#sum_list=[]
#print(sum_list)
#for i in range(len(cluster)):
 #   a=reshape(kuroster[0].z[i],(169,1))
  #  cosa=np.matmul(a,w1)
   # for (item1, item2) in zip(cosa, b1):
    #    sum_list.append(item1+item2)
    
    #arreglo=[]
    #arreglo.append(sum_list)
    
#sig= lambda t: 1/(1+exp(-t))
#sigmoid=np.vectorize(sig)
#tsig=sigmoid(arreglo)


#swish= lambda t: t/(1+exp(-t))
#switch=np.vectorize(swish)
#tswish=switch(arreglo)

#ufas=sum(multiply(tsig,tswish),1)

#for i in range(len(kuroster[0].vi[0])):
 #   kuroster[0].vi[0][i]=kuroster[0].vi[0][i]+ufas[0][i]
        
        
model = keras.Sequential(
    [
        keras.Input(shape=(169,1)),
        model.add(layers.Conv2D(169, 1, strides=2, activation="relu"))
        model.add(layers.Conv2D(169, 1, activation="relu"))
    ]
)  # No weights at this stage!

# At this point, you can't do this:
# model.weights

# You also can't do this:
# model.summary()

# Call the model on a test input
#x = tf.ones((1, 4))
#y = model(x)
print("Number of weights after calling the model:", len(model.weights)) 

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
for i in range(len(cluster)):
    for j in range(len(cluster)):
        model.fit(kuroster[i].z[j], epochs=10)
        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
        
