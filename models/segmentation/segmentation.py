#!/usr/bin/env python3

import numpy as np
import netCDF4 as nc
from os import listdir
from os.path import expanduser

inputfolder = "~/data/segmentation_input"
outputfolder = "~/data/segmentation_output"
# transformming "~" into "/home/$USER"
inputfolder = expanduser(inputfolder)
outputfolder = expanduser(outputfolder)

ncfns = [ fn for fn in listdir(inputfolder) if fn[-3:]=='.nc' ]
for ncfn in ncfns:
    pass
