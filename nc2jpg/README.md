# nc2jpg

This program exports a NetCDF file variable to a JPEG file (values are scaled to fit between 0-255). JPEG quality can be chosen. This code is meant to ease posterior creation of handmade masks for a NetCDF variable with an image editor like *GIMP*.


## Dependencies:

You need `Python 3+` and `pip 3+` in order to use this script.

| Python libraries needed |
| - |
| numpy |
| netcdf4 |
| pillow |

If your distribution still calls `python` as `python2`, instead of `python3` (you can tell that by sending `realpath $(which python)`), as debian and debian based distros do (Ubuntu and Linux Mint are examples), use `pip3` instead of `pip` to install these dependencies with:

Install them with:
```
pip install --user numpy netcdf4 pillow
# or, if  python3 isn't default,
pip3 install --user numpy scipy pillow
```


## USAGE

```sh
./nc2jpg.py INPUT_FILE_PATH.nc [-o OUTPUT_PATH] [-v VARIABLE] [-q QUALITY]
```

Input file argument:
  - `INPUT_FILE_PATH`: path of NetCDF file to be saved as picture. Needs to be the first argument given.

Arguments for options
  - `VARIABLE`: exact name of NetCDF file variable to be extracted.
  - `OUTPUT_PATH`: path to save picture. Can be either a complete filepath, or a filepath without extension (defaults to save picture in JPEG), or even a folder path to save file into (in this case, file is saved as "`INPUT_FILE_PATH`.jpg").
  - `QUALITY`: in case JPEG is your output format, then specify compression quality (between 0 and 100, lower `QUALITY` means lossier compression). This parameter is completely ignore in case another format is chosen.

If options are not chosen, nc2jpg will prompt you for them.


## contact

  - **Name**: Marcos Reinan de Assis Conceição.
  - **E-mail**: [marcosrdac@gmail.com](mailto:marcosrdac@gmail.com)
  - **GitHub**: [marcosrdac](github.com/marcosrdac)
  - **Website**: [marcosrdac.github.io](http://marcosrdac.github.io)
