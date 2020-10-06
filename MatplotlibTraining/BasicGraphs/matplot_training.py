import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

''' Plot first simple graph '''
x = [0, 1, 2, 3, 4]
y = [0, 2, 4, 6 ,8]

## resize graph
plt.figure(figsize=(5,3), dpi=120)     #dpi - pixels per inch, total size would be figsize times dpi (600 px / 360 px)

## plot lines
plt.plot(x, y, label='2x', color='lightgreen', linewidth=2, marker='*',\
        markersize=10, markeredgecolor='blue', linestyle='--')

### use shorhand notation of above line
#fmt = '[color][marker][line]'
#plt.plot(x, y, 'g*--', label='2x',markeredgecolor='blue' )

## add line number 2
x2 = np.arange(0,5, 0.5)       # [0.  0.5 1.  1.5 2.  2.5 3.  3.5 4.]
# plot part of the graph as solid line
plt.plot(x2[:5], x2[:5]**2, color='red', label='x^2')   # first 4 values
# plot the rest of the graph using a dashed line
plt.plot(x2[4:], x2[4:]**2, 'r--')   # rest of the values

## add graph title
plt.title("Our first graph", fontdict={'fontname': 'Comic Sans MS', 'fontsize': 20})

## X and Y labels
plt.xlabel("x axis", fontdict={'fontname': 'Comic Sans MS', 'fontsize': 10})
plt.ylabel("y axis", fontdict={'fontsize': 10})

## modify tickmarks
plt.xticks(x+[6])
#plt.yticks([*range(0,12,2)])        # * -> argument-unpacking operator

## add a legend
plt.legend()        # it uses labels previously used

## save graph, higher dpi gives higher resolution
plt.savefig('my_graph.png', dpi=300)



''' Bar Charts '''
## resize graph - open a new figure
plt.figure(figsize=(6,4), dpi=120)

## add X and Y values
labels = ['A', 'B', "C"]
values = [1,4,2]

## plot
bars = plt.bar(labels, values)

## change bars outlook

# # first approach
# bars[0].set_hatch('/')
# bars[1].set_hatch('o')
# bars[2].set_hatch('*')

# second approach
patterns=['/', 'o', '*']
for i in range(len(bars)):
        bars[i].set_hatch(patterns[i])          # w could go with 'for bar in bars' and use patterns.pop(0), no index needed


## show both graphs
plt.show()
