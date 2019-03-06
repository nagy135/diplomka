---
title:
- Progress presentation
subtitle:
- Algorithm development for the segmentation of astronomical images with unique features
author:
- Viktor Nagy
theme:
- Warsaw
---

# Space Debris

![Space debris sources](debris.png){#id .class width=100% }

# Space View

![Debris clusters (GEO, LEO)](spaceview.png){#id .class width=100% }

# GEO/LEO

## There are multiple orbital layers
+ LEO (Low Earth Orbit)
+ GEO (Geosynchronous Earth Orbit)
+ GNSS, GTO, Molniya

##
![Orbital layers](geoleo.png){#id .class height=35% }

# Change over time

![Amount of debris through time milestones](time.png){#id .class width=100% }

# AGO 70cm

## AGO 70 programs
+ Astrometry, surveys
+ Photometry, light curves
+ Photometry, colors

##
![AGO 70cm installation(left), mount(middle), primary mirror(right)](telescope.png){#id .class height=35% }

# Pipeline

![Astrometry pipeline](pipeline.png){#id .class height=100% }

# Tracking

## There are 2 types of tracking
+ Sidereal tracking
+ Object tracking

##
![Posible tracking methods, Sidereal tracking(left), Object tracking(right)](tracking.png){#id .class height=35% }

# Steps

+ Image capture
+ Image reduction
+ Sky background estimation/extraction
+ Objects search and centroiding
+ Star field identification
+ Astrometric reduction
+ Star Masking
+ Tracklet building
+ Object identification
+ Data format transformation
+ Output data redistribution

# Sky background estimation/subtraction

## Reasons/Causes
+ Moon light (global linear gradient)
+ Stars, Nebulas, Galaxies (local nonlinear gradients)
+ Hardware related reflexions

## Methods
+ Convolution with large median kernel (at least 25% of the size of image)
+ Sigma clipping


# Sky background estimation/subtraction

![original image(a), background(b), result median filtering(c), sigma clipping(d)](background.png){#id .class height=60% }

# Sky background estimation/subtraction

![Dumbell nebula M27, AGO 70cm telescope](nebula.png){#id .class height=60% }

# Sky background estimation results

![original image(left), estimated background(right)](8_image_rgb.png){#id .class height=60% }
![original image(left), estimated background(right)](8_extracted_rgb.png){#id .class height=60% }

# Object identification

![](noise.png){#id .class height=100% }

# Object identification

## Methods
+ PSF fitting
+ Edge detection
+ Barycenter positions

# PSF objects detected

![Psf fitting results](point_mesh_2.png){#id .class height=65% }
![Psf fitting results](point_mesh_3.png){#id .class height=65% }

# PSF fitting - trail

![Trail shown from 3d perspective](streak.png){#id .class height=60% }

# PSF fitting

![PSF fitting equations](streak_equation.png){#id .class height=90% }

# Output

![Final result, .cat file](cat.png){#id .class height=90% }

# Tools

## Python

+ Numpy
+ Astropy (fits files)
+ Scipy (convolve, fitting)
+ Matplotlib
+ OpenCV
+ Plotly
+ AstroImageJ

# Sources

+ Heiner Klinkrad - Space Debris, Models and Risk Analysis
+ Vladimir Kouprianov et. al. - Distinguishing features of CCD astrometry of faint GEO objects
+ J. Šilha et. al. - Slovakian Optical Sensor for HAMR Objects Cataloguing and Research
+ Jenni Virtanen et. al. - Streak detection and analysis pipeline for space-debris optical images
+ Peter Vereš et. al. - Improved Asteroid Astrometry and Photometry with Trail Fitting
+ Oddelenie astronómie a Astrofyziky - Fakulta matematiky fyziky a informatiky, UK
+ Edith Stöveken et. al. - Algorithms for the Optical Detection of Space Debris Objects

# The End

![](thanks.png){#id .class width=100% }
