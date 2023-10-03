# Imports
import os
import sys
import platform
import json
import numpy as np
import pandas as pd 
import math
import glob
#from json2html import *
import seaborn as sns
from astropy.io import ascii,fits
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import z_at_value
from astropy.cosmology import FlatLambdaCDM
from astropy.wcs import WCS

from astropy.stats import binom_conf_interval
from astropy.visualization import ZScaleInterval
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from photutils.aperture import CircularAperture

from scipy.interpolate import interp2d,RectBivariateSpline
import matplotlib
import matplotlib as mpl
import matplotlib.ticker as plticker
import matplotlib.patches as patches
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

# Use this version for non-interactive plots (easier scrolling of the notebook)
#%matplotlib inline

# Use this version if you want interactive plots
# %matplotlib notebook

# These gymnastics are needed to make the sizes of the figures
# be the same in both the inline and notebook versions
#%config InlineBackend.print_figure_kwargs = {'bbox_inches': None}

mpl.rcParams['savefig.dpi'] = 200
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams.update(matplotlib.rcParamsDefault)

# define cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)



# CHANGE THIS LINE TO POINT TO YOUR LOCAL VERSION
#root = './Data/'
os_check = platform.platform(terse=True)[:5]

if os_check == 'macOS':
    preamble = '/Users/lordereinion/'
else:
    preamble = '/home/six6ix6ix/'
root1 = f'{preamble}Dropbox/KU/Code/astro/GOGREEN_Data/Data/'
root2 = f'{preamble}Dropbox/KU/Code/astro/STARRED_Deconvolution/mos/'


# Load the .fits files 
data, header = fits.getdata(f'{root2}mos_HAWKIKs_3.fits', header=True)
psf, header2 = fits.getdata(f'{root2}mos_HAWKIKs.psf_3.fits', header=True)
noise, header3 = fits.getdata(f'{root2}mos_HAWKIKs.psf_3.weight.fits', header=True)

# Creates the directory to store the sliced images
#slice_dir = 'slices'
star_slice_dir = 'stars'
star_slice_psf_dir = 'stars_psf'
star_slice_noise_dir = 'stars_noise'
gal_slice_dir = 'galaxy'
gal_slice_psf_dir = 'galaxy_psf'
gal_slice_noise_dir = 'galaxy_noise'
slice_path_dir = f'{root2}slices'


# define paths for catalogue data

#version = 'DR1
#version = 'v2.0'
#dirr  =  root + version + '/'
dirr  = root1
specdir = f'{dirr}SPECTROSCOPY/'
catdir = f'{dirr}CATS/'
photdir = f'{dirr}PHOTOMETRY/'
imdir = f'{photdir}IMAGES/'
oneddir = f'{specdir}OneD/'
twoddir = f'{specdir}TwoD/'

# cluster data

clusters = f'{catdir}Clusters.fits'

# read in fits data table with astropy.table.Table and immediately convert to pandas Dataframe
cluster_table = Table( fits.getdata( clusters ) ).to_pandas() 
cluster_table['cluster'] = cluster_table['cluster'].str.rstrip().values # remove unnecessary spaces
#display(cluster_table.columns)
#display(cluster_table)
#display(len(cluster_table.columns))
#display(len(cluster_table))


# photometric data
photfile = f'{catdir}Photo.fits'
phot_table = Table(fits.getdata(photfile)).to_pandas()
#display(phot_table.columns)
#display(phot_table)

#display(len(phot_table.columns))
#display(len(phot_table))

# redshift data
zcatfile = f'{catdir}Redshift_catalogue.fits'

# read in fits data table with astropy.table.Table and immediately convert to pandas Dataframe
redshift_table = Table( fits.getdata( zcatfile ) ).to_pandas() 
redshift_table['Cluster'] = redshift_table['Cluster'].str.rstrip().values # remove unnecessary spaces
#display(redshift_table.columns)
#display(redshift_table)

#display(len(redshift_table.columns))
#display(len(redshift_table))

# Merging the spectroscopic data with the photometry data

cluster, cluster_id, stelmass_cat = cluster_table[['cluster', 'cluster_id', 'stelmass_cat']].values.T

stellpops_table = pd.DataFrame()

for idx,clust in enumerate(cluster):
    if ((cluster_id[idx] < 13) & (cluster_id[idx] != 7)):
        if (cluster_id[idx] < 4):
            clust = clust.replace('SPT','SPTCL-')
        elif (cluster_id[idx] >= 4):
            clust = clust.replace('SpARCS','SpARCS-')
        # constructs filename to be used
        specz_matchfile = f'compilation_{clust}.dat'
        specz_matchpath = f'{photdir}SPECZ_MATCHED/{specz_matchfile}'
        # select columns from spectrscopic data
        specz_matchcols = ['PHOTCATID', 'spec_z','source', 'quality', 'distance', 'altspec_z', 'altdistance', 'altsource', 'altquality', 'SPECID' ]
        speczmatch_table = pd.read_table(specz_matchpath,comment='#',sep='\s+',names=specz_matchcols)
        # stellar mass data
        stelmass_path = f'{photdir}STELMASS_CATS/{stelmass_cat[idx].rstrip()}'
        # columns for stellar mass data
        stelmass_cols = ['pID', 'z', 'ltau', 'metal', 'lage', 'Av', 'lmass', 'lsfr', 'lssfr', 'la2t', 'chi2']
        stelmass_table = pd.read_table(stelmass_path,comment='#',sep='\s+',names=stelmass_cols)
        pID = stelmass_table[['pID']].values.T

        surv = 1
        cPHOTpre = f'{surv}{cluster_id[idx]:0>2d}'
        cPHOTID = np.zeros(len(stelmass_table['pID']),dtype=np.int64)
        
        # Create photometric id cPHOTOID
        #this loop builds the cPHOTID for every galaxy
        for igal,id in enumerate(stelmass_table['pID']):
            idstr = f"{id:0>6d}"
            cPHOTID[igal] = int(cPHOTpre + idstr)
        
        #add the cPHOTID as a new table column
        stelmass_table['cPHOTID'] = cPHOTID
        
        #this only ouputs columns with names different than those in the speczmatch table.  
        #cols_to_use = stelmass_table.columns.difference(speczmatch_table.columns).tolist()

        #instead specify columns by hand
        cols_to_use = ['cPHOTID','chi2','la2t','lage','lmass','lsfr','lssfr','ltau','metal','Av','z']

        #takes the "cols_to_use" columns from the STELMASS table and join them onto the 
        #SPECZMATCh table matching adjoining values on the same row as the tables are row-matched.
        matched_table = pd.merge(speczmatch_table, stelmass_table[cols_to_use], how='left', \
                         left_index=True, right_index=True)
        
        #Also merge in rest-frame colors - only considering GOGREEN clusters again
        rf_path = photdir + 'RESTFRAME_COLOURS/RESTFRAME_MASTER_' + clust + '_indivredshifts.cat'
        rf_cols = ['PHOTID', 'REDSHIFTUSED', 'FUV', 'NUV', 'U', 'B', 'V', 'R', 'I', 'Z', 'J', 'H', 'K', 'sdssu', 'sdssg', 'sdssr', 'sdssi', 'sdssz', 'UMINV', 'VMINJ']
        rf_table = pd.read_table(rf_path,comment='#',sep='\s+',names=rf_cols)
        rf_table['cPHOTID'] = cPHOTID

        cols_to_use = ['NUV','FUV','U','V']
        matched_table = pd.merge(matched_table, rf_table[cols_to_use], how='left', \
                                 left_index=True, right_index=True)

        #append the information to the stellar populations table
        
        #stellpops_table = stellpops_table.append(matched_table,ignore_index=True)
        stellpops_table = pd.concat([stellpops_table, matched_table], ignore_index=True) 
## Add NUVMINV field
stellpops_table['NUVMINV'] = -2.5 * np.log10(stellpops_table['NUV']/stellpops_table['V'])
matched_table['NUVMINV'] = -2.5 * np.log10(matched_table['NUV']/matched_table['V'])
#display(matched_table.truncate(253, 253))
#display(len(matched_table.columns))
#display(len(matched_table))


# Merge photo and spec tables to return a table that has photometric information (if available) 
# for every object in the redshift catalogue

# this way avoids duplicate columns (ie dont need to specify suffixes)
merge_col = ['SPECID']

#this only ouputs columns with names different than those in the redshift table.  
#Make sure that SPECID is added back in as we will match on that field
cols_to_use = phot_table.columns.difference(redshift_table.columns).tolist() + merge_col

#takes the "cols_to_use" columns from the photometric table and join them onto the 
#redshift table matching on the SPECID field.
matched_table = pd.merge(redshift_table, phot_table[cols_to_use], how='left', \
                         left_on=['SPECID'], right_on=merge_col )

#convert cPHOTID from float to int, because it was somehow converted to a float in the previous merge
cPHOTID = matched_table['cPHOTID'].values.T
cPHOTID = cPHOTID = cPHOTID.astype(int)
matched_table['cPHOTID'] = cPHOTID

#display(matched_table)
merge_col = ['cluster']
cols_to_use = cluster_table.columns.difference(matched_table.columns).tolist() 
#matched_table = pd.merge(matched_table, cluster_table[cols_to_use], how='left', \
#                         left_on=['Cluster'], right_on=merge_col )

#now match info about the clusters to the spectroscopic table.  
# Here attach suffix _c to distinguish between galaxy values (Redshift) and cluster values (Redshift_c)
matched_table = pd.merge(matched_table, cluster_table, how='left', \
                         left_on=['Cluster'], right_on=merge_col, suffixes=['','_c'] )

#now merge spectroscopic catalog with stellar populations parameters from above
###FOR JACOB - Need to make sure that you add the rest-frame clors to the "cols_to_use" list.
cols_to_use = ['ltau', 'metal', 'lage', 'Av', 'lmass', 'lsfr', 'lssfr', 'la2t', 'chi2','SPECID','NUV','FUV','U', 'V', 'NUVMINV']
matched_table = pd.merge(matched_table,stellpops_table[cols_to_use],how='left', \
                         left_on=['SPECID'], right_on=['SPECID'])

#matched_table_p = pd.merge(matched_table,stellpops_table[cols_to_use], on= 'SPECID')


#display(matched_table.columns.values)
#display(matched_table)

# Merge photo and spec tables to return a table that has photometric information (if available) 
# for every object in the redshift catalogue

# this way avoids duplicate columns (ie dont need to specify suffixes)
merge_col = ['cPHOTID']

#this only ouputs columns with names different than those in the redshift table.  
#Make sure that SPECID is added back in as we will match on that field

cols_to_use = stellpops_table.columns.difference(phot_table.columns).tolist() + merge_col

cols_to_use = ['ltau', 'metal', 'lage', 'Av', 'lmass', 'lsfr', 'lssfr', 'la2t', 'chi2','cPHOTID','FUV','NUV','U','V', 'NUVMINV']

#takes the "cols_to_use" columns from the photometric table and join them onto the 
#redshift table matching on the SPECID field.
photpops_table = pd.merge(phot_table, stellpops_table[cols_to_use], how='left', \
                         left_on=['cPHOTID'], right_on=merge_col )
#convert cPHOTID from float to int, because it was somehow converted to a float in the previous merge
cPHOTID = matched_table['cPHOTID'].values.T
#cPHOTID = pd.to_numeric(cPHOTID)
cPHOTID = cPHOTID.astype(int)
matched_table['cPHOTID'] = cPHOTID


merge_col = ['cluster']
cols_to_use = cluster_table.columns.difference(photpops_table.columns).tolist()
phot_table = pd.merge(phot_table, cluster_table[cols_to_use], how='left', \
                         left_on=['Cluster'], right_on=merge_col, suffixes=['','_c'] )
photpops_table = pd.merge(photpops_table, cluster_table, how='left', \
                         left_on=['Cluster'], right_on=merge_col, suffixes=['','_c'] )

spec_phot_table=photpops_table.merge(matched_table, how='outer')
spec_phot_table = spec_phot_table[~spec_phot_table['NUVMINV'].isna() ]

# Calculate RA and DEC in from degrees into radians, then convert to megaparsecs
spec_phot_table['delta_RA'] = np.deg2rad(np.cos(spec_phot_table['DEC_Best'])*(abs(spec_phot_table['RA_Best'] - spec_phot_table['RA(J2000)'])))
#matched_table['delta_RA'] = np.deg2rad(np.cos(matched_table['DEC(J2000)'])*(abs(matched_table['RA_Best'] - matched_table['RA(J2000)'])))
ang_dd = cosmo.angular_diameter_distance(1.5)
spec_phot_table['delta_DEC'] = np.deg2rad(abs(spec_phot_table['DEC_Best'] - spec_phot_table['DEC(J2000)']))
spec_phot_table['delta_theta'] = ang_dd*np.deg2rad((np.sqrt(spec_phot_table['delta_RA']**2 + spec_phot_table['delta_DEC']**2)))
# grabs only the data from SPT2106
#spt_spec_phot = spec_phot_table[spec_phot_table['Cluster']=='SPT2106']
#display(spec_phot_table)
#display(spt_spec_phot)

# This is the WCS object I created to convert the pixel data to RA and DEC in ICRS coordinates
wcs = WCS(header)

spec_phot_table.to_csv('spec_phot_table.csv')
















