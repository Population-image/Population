import sys
import os

if (sys.platform == "win32" or sys.platform == "win64"):
	PathPop= "D:\Users/vtariel/Desktop/ANV/Population-build/"
	sys.path.append(PathPop)
	from population import * #uncomment this line for windoww
else:
	PathPop= "/home/vincent/Population-build/"
	sys.path.append(PathPop)
	from population import * #uncomment this line for linux



try:
	img =Mat2UI8()#2d grey-level image object
	img.load(PathPop+"/image/iex.png")#replace this path by those on your computer
	#img.display("initial")
	proc = Processing()
	img = proc.median(img,3,2)#filtering
	threshold = proc.thresholdOtsuMethod(img)#threshold segmentation with OtsuMethod
	threshold.save("iexseg.png")
	visu = Visualization()
	color = visu.labelForeground(threshold,img)#Visual validation
	color.display()
except pexception, e:
	e.display()
