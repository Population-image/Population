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
    scene.display()
except pexception, e:
    e.display()
