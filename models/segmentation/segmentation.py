#!/usr/bin/env python3

import numpy as np
import netCDF4 as nc
from os import listdir, makedirs
from os.path import expanduser, isdir

inputfolder = "~/data/3_original_and_mask"
outputfolder = "~/data/4_original_and_segmentation"
# transformming "~" into "/home/$USER"
inputfolder = expanduser(inputfolder)
outputfolder = expanduser(outputfolder)
for folder in [inputfolder, outputfolder]:
    makedirs(folder, exist_ok=True)

ncfns = [ fn for fn in listdir(inputfolder) if fn[-3:]=='.nc' ]
for ncfn in ncfns:
    pass
