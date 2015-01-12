import sys
import os
PathPop= ""
if os.path.isfile(PathPop+"population.py")==0:
    print "set the variable PathPop to the path where you compile population, for instance D:\Users/vtariel/Desktop/ANV/Population-build/. This folder must contain population.py"
    sys.exit(-1)
sys.path.append(PathPop)
from population import * 
	

img = Mat3UI8()
img.load(PathPop+"/image/rock3d.pgm")
#	img.loadFromDirectory("/home/vincent/Desktop/WorkSegmentation/sand/","500-755","pgm") #to load a stack of images
img = img(Vec3I32(0,0,0),Vec3I32(64,64,64))
img.display()
pde = PDE()
imgfilter= pde.nonLinearAnisotropicDiffusionDericheFast(img)
proc = Processing()
grain= proc.thresholdOtsuMethod(imgfilter)
grain.display()

visu = Visualization()
color= visu.labelForegroundBoundary(grain,img)
color.display()
grain.save(PathPop+"/image/grain.pgm")

scene = Scene3d()
visu.marchingCube(scene,grain)
visu.lineCube(scene,img)
scene.display()

