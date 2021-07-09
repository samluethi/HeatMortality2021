#!/bin/sh

###### HEADER - Get ZRH TAS Data ############################################
# Author:  Samuel Luethi (samuel.luethi@usys.ethz.ch
# Date:    18.06.2021
# Purpose: Get TAS data from CESM12-LE
##############################################################################

## Go to the correct data location:
cd /net/bio/climphys/fischeer/CMIP5/EXTREMES/CESM12-LE/

## define out dir
outdir=/home/saluethi/heat/CESM12-LE_ZRH/

## Loop over the ensembles:
for ens_i in {1..83} #1..83
do 
  echo "Working on ensemble: $ens_i"
  
  if [ "$ens_i" -le "9" ]; then
    ens_iz=0$ens_i
  else
    ens_iz=$ens_i 
  fi
  
  ## Define which time period is covered by each member:
  indmax_1850=20
  if [ "$ens_i" -le "$indmax_1850" ]; then
    time_period=1850-2099
  else
    time_period=1940-2099
  fi
  
  ## get in/out file name:
  file_tas_in=tas_CESM12-LE_historical_r${ens_i}i1p1_${time_period}.nc
  file_tas_out=${outdir}ZRH_tas_CESM12-LE_historical_r${ens_i}i1p1_${time_period}.nc
  
  #echo "Indir: ${file_tas_in}"
  #echo "Indir: ${file_tas_out}"

  cdo -s -sellonlatbox,7.5,9.5,46.3,48.3 $file_tas_in $file_tas_out

done

exit 0












