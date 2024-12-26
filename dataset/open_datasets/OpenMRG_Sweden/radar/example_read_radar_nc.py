# -*- coding: utf-8 -*-
"""
@author: Remco van de Beek
Example script to read radar.nc file from OpenMRG dataset
"""
import matplotlib.pyplot as plt
import numpy as np
import netCDF4
import datetime

#input file
inFile='/data/open_data/radar/radar.nc'

#read radar file and variables
RadarDataset = netCDF4.Dataset(inFile) #Prepare reading data
print(RadarDataset.variables) #print variables 
theRadar = RadarDataset.variables['data'] #Read data variable
print(theRadar.shape) #print dimensions
dtime=netCDF4.num2date(RadarDataset.variables['time'][:],RadarDataset.variables['time'].units) #create datetime object

#Plot a time step 
selTime=datetime.datetime(2015, 7, 28, 16, 15) #select time step (Y,m,d,H,M)
selTimeStr=selTime.strftime("%Y-%m-%d %H:%M:%S") #Format datetime
selIndex=(dtime == selTime).argmax() #Get index of Time dimension for selectied time step
print(dtime[selIndex]) #Check selection
currentData=np.array(theRadar[selIndex,:,:]) #Select the x,y grid for current time step
plt.imshow(currentData) #Plot the data
plt.title(selTimeStr)