# Stat4Ci-Toolbox
 
Functions to analyze Classification Images

 DEMOSTAT4CI illustrates the usage of the Stat2Ci toolbox on two classification images from:
		Gosselin, F. & Schyns, P. G. (2001).  Bubbles: A technique to reveal the use of information 
 		in recognition. Vision Research, 41, 2261-2271.
 	The Stat4Ci toolbox allows to perform the Pixel and the Cluster tests, both based on Random Field 
 	Theory. These tests are easy to apply, requiring a mere four pieces of information; and they 
 	typically produce statistical thresholds (or p-values) lower than the standard Bonferroni correction.
 
	An excellent non-technical reference is:  		K. J. Worsley (1996) the geometry of random image. Chance, 9, 27-40.
 
	We borrowed from several sources: the STAT_THRESHOLD function was written by Keith Worsley for the fmristat toolbox
 	(http://www.math.mcgill.ca/~keith/fmristat); and our DISPLAYCI function calls many functions from the 
	Image Processing toolbox.
 
 The Stat4Ci toolbox is free if you use  it in your research, please, cite us:
	Chauvin, A., Worsley, K. J., Schyns, P. G., Arguin, M. & Gosselin, F. (2005).  Accurate statistical test for smooth classification images; Journal of Vision, 5(1), doi:https://doi.org/10.1167/5.9.1
 
 Alan Chauvin (Alan.Chauvin@univ-grenoble-alpes.fr) & Frédéric Gosselin (frederic.gosselin@umontreal.ca), 20/08/2004