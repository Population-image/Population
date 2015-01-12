import sys
import os
PathPop= ""
if os.path.isfile(PathPop+"population.py")==0:
    print "set the variable PathPop to the path where you compile population, for instance D:\Users/vtariel/Desktop/ANV/Population-build/. This folder must contain population.py"
    sys.exit(-1)
sys.path.append(PathPop)
from population import * 
    

img = Mat2UI8()
img.load(PathPop+"/image/iex.png")
proc = Processing()
filter = proc.smoothDeriche(img,1)
filter = proc.dynamic(filter,40)
minima = proc.minimaRegional(filter)
visu = Visualization()
#	visu.labelForeground(minima,img).display()
water = proc.watershedBoundary(minima,filter,1)
boundary = proc.threshold(water,0,0)#the boundary label is 0
#boundary.display("boundary",true,false)
minima = proc.labelMerge(Mat2UI32(boundary),minima)#the pixel type of the image must be the same f.
gradient = proc.gradientMagnitudeDeriche(img,1)
water = proc.watershed(minima,gradient)
visu.labelForeground(water,img).display()

