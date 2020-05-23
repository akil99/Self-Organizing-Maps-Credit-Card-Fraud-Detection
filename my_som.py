import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, plot, show, pcolor, colorbar

df = pd.read_csv('Credit_Card_Applications.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#normalization
scaler = MinMaxScaler(feature_range= (0,1))
X = scaler.fit_transform(X)

som = MiniSom(x= 10, y= 10, input_len= 15) #10 x 10 SOM
som.random_weights_init(X)
som.train_random(X, num_iteration= 100)

#SOM pLotting
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
#green squares - customers approved by bank
#red circles - customers rejected by bank
for i, x in enumerate(X):
    w = som.winner(x)   #w - winning neuron(BMU)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

mappings = som.win_map(X) 
#mappings - a dictionary containing the winning node and neighbouring nodes in the neighbourhood (sigma = 1)
frauds = np.concatenate((mappings[(4,1)], mappings[(6,1)]), axis = 0)
#Outling winning nodes are potential frauds
#(4,1) and (6,1) nodes respresent the fradulent customers
frauds = scaler.inverse_transform(frauds)
#frauds contain the list of fradulent customer's details
