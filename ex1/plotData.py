'''
Created on 10 feb. 2017

@author: fara
'''

def plotData(ax,x, y):
    
    '''PLOTDATA Plots the data points x and y into a new figure 
       PLOTDATA(x,y) plots the data points and gives the figure axes labels of
       population and profit.
    '''
    
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the training data into a figure using the 
    #               "figure" and "plot" commands. Set the axes labels using
    #               the "xlabel" and "ylabel" commands. Assume the 
    #               population and revenue data have been passed in
    #               as the x and y arguments of this function.  
    #
    # Hint: You can use the 'rx' option with plot to have the markers
    #       appear as red crosses. Furthermore, you can make the
    #       markers larger by using plot(..., 'rx', 'MarkerSize', 10);
    
    #Generate the plot
    ax.plot(x,y,linestyle='None',marker="x",color="r",label='Training Data')
    #Settup the legend
    ax.legend(loc='upper right', fontsize='x-small', numpoints=1)
    #Define the axis labels   
    ax.set_ylabel('Profit in $10,000s')
    ax.set_xlabel('Population of City in 10,000s')
    #Define axis limits
    ax.axis([4, 24, -5, 25])    
    
    
    
    
    # ============================================================
    
  