# 2021-11-07 Victor Näslund <victor.naslund@smhi.se>
# Example code to read netcdf data

# Run with python3 <this_file>

# Useful tools:
# ncdump -h <my_netcdf.nc> # Show netcdf metadata
# cdo info -seltimestep,1/50 <my_netcdf.nc> # Show info about first 50 timesteps

# More documentation
# https://unidata.github.io/netcdf4-python/
# https://numpy.org/doc/stable/reference/
#
#For windows users the program Panoply is useful for easy visualisation of netcdf data
#https://www.giss.nasa.gov/tools/panoply/
# 

# Import libraries
import numpy as np
import netCDF4

# Open the netcdf files
nc = netCDF4.Dataset("cml.nc", "r")

print ("Show some info for variable 'tsl'")
print (nc.variables["tsl"])

# Show the tsl variable data shape
print ("The shape of the variable 'tsl' is ")
print (nc.variables["tsl"].shape)


# Get the (entire tsl variable from the netCDF file. Note: a lot of data!
#data = np.array(nc.variables["tsl"][:, :, :])

# Get the first 50 timesteps for the rsl variable
data = np.array(nc.variables["rsl"][0:50, :])


# Print all the data in the first timestep
print ("Showing all rsl data in the first timestep")
print (data[0, :])


print ("Get rsl data for first 50 timesteps for the third sublink")
print (np.array(nc.variables["rsl"][0:50, 2]))

print ("'contact' attribute value is ")
print (nc.getncattr("contact"))

print ("The 'units' attribute for variable 'tsl' is ")
print (nc.variables["tsl"].getncattr("units"))

print ("Get the actual value for the time variable for the first timestep")
print ("Note that the value is '" + nc.variables["time"].getncattr("units") + "'")
print (nc.variables["time"][0])

print ()
print ("Therefore the time value for the next timestep is 10 more since each timestep is 10 seconds")
print (nc.variables["time"][1])

print()
print ("The missing value in tsl and rsl is 1e10 in the netcdf so we need to set it to np.nan before using the data")
data = np.array(nc.variables["tsl"][0:50,2])
data[data == 1e10] = np.nan

