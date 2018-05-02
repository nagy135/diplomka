=head1 NAME

Algorithm development for the segmentation of astronomical images with unique features

=head1 DESCRIPTION

During the astronomical observations images are acquired in so-called Flexible Image Transport System (FITS) format. This images contain signal from various sources, starting from electronic noise and readout noise, going trough sky background signal to object images such as stars or asteroids. On an typical astronomical image the stars and asteroids usually appear as point-like objects which can be described by the Point-Spread Function (PSF). This is not the case for the observations of space debris objects such as fragmentation debris, defunct satellites and upper stages. These objects move comparable faster than the stars in the background which leads to FITS images which have present trail-like objects. In case that sidereal tracking is used during the observations, stars are being tracked by the telescope, the space debris object appears as an trail and stars as points. In case that debris tracking is used the debris appears as a point and stars as a trails with the same length and direction on the image. 
One of the tasks for the possible candidate will be to review the existing algorithms for the image segmentation procedures currently used in the astronomical community for space debris observations. Depending on the review selected will the best algorithm which will be then written by the candidate in a testing environment and then will be tested on the real observations acquired by the optical systems currently present at the Astronomical and geophysical observatory in Modra. Algorithm will have to identify all image objects above defined thershold (Signal-to-Noise Ratio, SNR, to be defined during the work), for each image object extracted will be the position on the CCD frame, as well the total intensity. The algorithm efficiency will be investigated by comparing the results with the ground-truth extracted for the known objects, such as space debris or asteroids which will be delivered to the candidate.

=head1 GOALS

Development of an algorithm which will identify and extract positions of any features present on a single astronomical image acquired during the astronomical observations to space debris objects such as fragments, defunct satellites and upper stages. 

=head1 RELEVANT LITERATURE

E. Stöveken, T. Schildknecht, Algorithms for the Optical Detection of Space Debris Objects, Proceedings of the 4th European Conference on Space Debris (ESA SP-587). 18-20 April 2005, ESA/ESOC, Darmstadt, Germany. Editor: D. Danesy., p.637.
V. Kouprianov, Distinguishing features of CCD astrometry of faint GEO objects, Advances in Space Research, Volume 41, Issue 7, 2008, Pages 1029-1038, ISSN 0273-1177, http://dx.doi.org/10.1016/j.asr.2007.04.033. (http://www.sciencedirect.com/science/article/pii/S0273117707003699)
B. Flewelling, Computer Vision Techniques Applied to Space Object Detect, Track, ID, Characterize, Proceedings of the Advanced Maui Optical and Space Surveillance Technologies Conference, held in Wailea, Maui, Hawaii, September 9-12, 2014, Ed.: S. Ryan, The Maui Economic Development Board, id.E69.
