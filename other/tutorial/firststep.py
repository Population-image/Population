import sys
import os
PathPop= ""
if os.path.isfile(PathPop+"population.py")==0:
    print "set the variable PathPop to the path where you compile population, for instance D:\Users/vtariel/Desktop/ANV/Population-build/. This folder must contain population.py"
    sys.exit(-1)
sys.path.append(PathPop)
from population import * 

img =Mat2UI8()#2d grey-level image object
img.load(PathPop+"image/iex.png")#replace this path by those on your computer
img.display("initial")
pde = PDE()
img = pde.nonLinearAnisotropicDiffusion(img)
proc = Processing()
threshold = proc.thresholdOtsuMethod(img)#threshold segmentation with OtsuMethod
threshold.save("iexseg.png")
visu = Visualization()
color = visu.labelForeground(threshold,img)#Visual validation
color.display()

