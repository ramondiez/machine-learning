'''
Created on 24 feb. 2017

@author: fara
'''
from matplotlib import use

#use('TkAgg')
from __builtin__ import int
import matplotlib.pyplot as plt
import numpy as np
from show import show

def displayData(X):
    '''
    %DISPLAYDATA Display 2D data in a nice grid
    %   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    %   stored in X in a nice grid. It returns the figure handle h and the 
    %   displayed array if requested.
    '''
    #np.set_printoptions(threshold=np.inf)   
    #np.set_printoptions(precision=20)  
    
    m,n = X.shape
    show(X.shape)    
    example_width = int(round(np.sqrt(n)))    
    example_height = int((n / example_width))
    
    # Compute number of items to display
    display_rows = np.floor(np.sqrt(m))   
    display_cols = np.ceil(m / display_rows)
    
    
    # Between images padding
    pad = 1
    
    # Setup blank display
    x = pad + display_rows * (example_height + pad)
    y = pad + display_cols * (example_width + pad)    
    
    #display_array = np.ones((pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)))
    display_array = - np.ones((x.astype(np.int),y.astype(np.int)),dtype=np.float64)
    
    #show(display_array)
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in np.arange(display_rows):
        for i in np.arange(display_cols):
            if curr_ex > m:
                break                                    
            # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex, :]))           
            rows = [pad + (j * (example_height + pad)) + x for x in np.arange(example_height+1)]            
            cols = [pad + (i * (example_width + pad))  + x for x in np.arange(example_width+1)]
            display_array[int(min(rows)):int(max(rows)), int(min(cols)):int(max(cols))] = \
            X[curr_ex, :].reshape(example_height, example_width) / max_val                
            curr_ex +=1   
        if curr_ex > m: 
            break 
        
    
    #Set plot in interactive mode
    plt.ion() 
    # Display Image   
    plt.imshow(display_array.T)
    plt.set_cmap('gray')
    
    # Do not show axis
    plt.axis('off')
    plt.show()
    plt.pause(0.0001)
    raw_input("Program paused. Press Enter to continue.")
    plt.close()