import sys
import os
PathPop= ""
if os.path.isfile(PathPop+"population.py")==0:
    print "set the variable PathPop to the path where you compile population, for instance D:\Users/vtariel/Desktop/ANV/Population-build/. This folder must contain population.py"
    sys.exit(-1)
sys.path.append(PathPop)
from population import * 
    

def visu2DSlice() :
    img = Mat3UI8(PathPop+"/image/rock3d.pgm")
    img.display()#use the arrows  to move in z-axis

def visu3DCube():
    scene = Scene3d() 
    img = Mat3UI8(PathPop+"/image/rock3d.pgm")
    visu = Visualization()
    visu.cubeExtruded(scene,img)#add the cube surfaces to the scene
    visu.lineCube(scene,img)#add the border red lines to the scene to the scene
    scene.display(0)#display the scene
    waitKey()

def visu3DSlice():
    scene = Scene3d() 
    img = Mat3UI8(PathPop+"/image/rock3d.pgm")
    visu = Visualization()
    visu.plane(scene,img,50,2)
    visu.plane(scene,img,50,1)
    visu.plane(scene,img,200,0)
    visu.lineCube(scene,img)
    scene.display(0)
    waitKey()

def visu3DMarchingCube():
    img = Mat3UI8()
    img.load(PathPop+"/image/rock3d.pgm")
    img = img(Vec3I32(0,0,0),Vec3I32(64,64,64))
    pde = PDE()
    imgfilter= pde.nonLinearAnisotropicDiffusionDericheFast(img)
    app=Application()
    #app.thresholdSelection(imgfilter)
    proc = Processing()
    grain= proc.threshold(imgfilter,155)
    oil = proc.threshold(imgfilter,70,110)
    oil = proc.openingRegionGrowing(oil,2)
    air = proc.threshold(imgfilter,0,40)
    seed = proc.labelMerge(grain,oil)
    seed = proc.labelMerge(seed,air)
    visu =Visualization()
    #visu.labelForeground(seed,imgfilter).display()
    gradient = proc.gradientMagnitudeDeriche(img,1.5)
    water = proc.watershed(seed,gradient)
    grain = proc.labelFromSingleSeed(water,grain)
    grain=grain/2
    oil = proc.labelFromSingleSeed(water,oil)
    oil = oil/4
    scene = Scene3d()
    visu.marchingCube(scene,grain)
    visu.marchingCube(scene,oil)
    visu.lineCube(scene,img)
    scene.display(0)
    waitKey()
    
def visu3DCubeExtrudedWithAxis():
    scene = Scene3d() 
    img = Mat3UI8()
    img.load(PathPop+"/image/rock3d.pgm")
    img = img(Vec3I32(0,0,0),Vec3I32(64,64,64))
    extruded = Mat3UI8(img.getDomain())
    radius=img.sizeI()/2
    center = Vec3I32(img.getDomain()/2)
    it = img.getIteratorEDomain()
    while it.next():
        if (it.x()-center).norm(2)<radius:
            extruded.setValue(it.x(),100)
    visu = Visualization()
    visu.cubeExtruded(scene,img,extruded)#add the cube surfaces to the scene
    visu.lineCube(scene,img)#add the border red lines to the scene to the scene
    scene.display(0)#display the scene
    waitKey()
        
  
visu2DSlice()
visu3DCube()
visu3DSlice()
visu3DMarchingCube()
visu3DCubeExtrudedWithAxis()

