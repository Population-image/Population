import sys
import os
PathPop= os.getcwd()+"/../../"

sys.path.append(PathPop+"/lib")
if (sys.platform == "win32" or sys.platform == "win64"):
    from populationpythonwin32 import * #uncomment this line for windoww
else:
    from population import * #uncomment this line for linux

#### class member ###
def classmember() :
	img = Mat2UI8(PathPop+"/image/Lena.bmp")
	img.display()#display the image
	img.display("lena",False,False)#display the image with lena title without stopping the execution without resizing the image 


#### display class ###
def classdisplay() :
	img = Mat2UI8(PathPop+"/image/Lena.bmp")
	disp =  MatNDisplay()#class to display the image
	#execute and display the result in the object window disp while this windows is not closed
	i =1
	while True :
		proc = Processing()
		img = proc.erosionRegionGrowing(img,i,2)
		img = proc.dilationRegionGrowing(img,i,2)
		i = i+1
		disp.display(img)
		if disp.is_closed() is True :
			break


#### Visualization algorithms ###
def visualizealgorithm() :
	# label image in foreground of a grey(color) imageto check segmentation or seed localization  
	img = Mat2UI8(PathPop+"/image/Lena.bmp")
	proc = Processing()
	thre = proc.thresholdOtsuMethod(img)
	visu = Visualization()
	foreground = visu.labelForeground (thre,img,0.7)
	foreground.display()
	
	# display each label with a random colour
	d = DistributionPoisson(0.001)
	field = Mat2UI32(512,512)#realisation of a discrete poisson field 
	it = field.getIteratorEDomain()
	label = 1
	while it.next() is True:
		if d.randomVariable() != 0:
			field.setValue(it.x(),label)
			label=label+1 
	proc = Processing()
	field = proc.voronoiTesselationEuclidean(field)#voronoi tesselation with  2-norm
	visu = Visualization()
	visu.labelToRandomRGB(field).display()
	
	
try:	
	classmember()
	classdisplay()
	visualizealgorithm()
except pexception, e:
	e.display()
