import sys
import os
PathPop= os.getcwd()+"/../../"

sys.path.append(PathPop+"/lib")
if (sys.platform == "win32" or sys.platform == "win64"):
    from populationpythonwin32 import * #uncomment this line for windoww
else:
    from population import * #uncomment this line for linux
    
try:
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
except pexception, e:
	e.display()
