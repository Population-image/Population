import sys
import os
PathPop= os.getcwd()+"/../../"

sys.path.append(PathPop+"/lib")
if (sys.platform == "win32" or sys.platform == "win64"):
	from populationpythonwin32 import * #uncomment this line for windoww
else:
	from population import * #uncomment this line for linux
	
try:
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
except pexception, e:
	e.display()
