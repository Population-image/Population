import sys
import os
PathPop= "/home/vincent/DEV2/Population-build/"#replace by yours
if os.path.isfile(PathPop+"population.py")==0:
    print "set the variable PathPop to the path where you compile population, for instance D:\Users/vtariel/Desktop/ANV/Population-build/. This folder must contain population.py"
    sys.exit(-1)
sys.path.append(PathPop)
from population import * 


import math

def testUniformPoissonPointProcess2D() :
	domain = Vec2F32(512,512)#2d field domain
	lambdafield = 0.001#parameter of the Poisson point process
	random = RandomGeometry()
	grain = random.poissonPointProcess(domain,lambdafield)#generate the 2d Poisson point process
	d = DistributionDirac(2)#because the Poisson point process has a surface equal to 0, we associate each point with mono-disperse sphere to display the result
	random.sphere(grain,d)
	img = random.continuousToDiscrete(grain)
	img.display()
def testUniformPoissonPointProcess3D() :
	domain = Vec3F32(100,100,100)#2d field domain
	lambdafield = 0.0001 #parameter of the Poisson point process
	random = RandomGeometry()
	grain = random.poissonPointProcess(domain,lambdafield)#generate the 2d Poisson point process
	d = DistributionDirac(3)#because the Poisson point process has a volume equal to 0, we associate each point with mono-disperse sphere to display the result
	print lambdafield
	random.sphere(grain,d)
	img = random.continuousToDiscrete(grain)
	print lambdafield
	scene=Scene3d()
	visu = Visualization()
	visu.marchingCube(scene,img)
	visu.lineCube(scene,img)
	print lambdafield
	scene.display()
	
def testNonUniformPoissonPointProcess2D() :
	img = Mat2UI8(PathPop+"/image/Lena.bmp")
	img = Mat2F32(img)
	img=img/255.
	img=img*0.05
	random = RandomGeometry()
	grain = random.poissonPointProcessNonUniform(img)#generate the 2d Poisson point process
	d = DistributionDirac(4)#because the Poisson point process has a surface equal to 0, we associate each point with mono-disperse sphere to display the result
	random.sphere(grain,d)
	img = random.continuousToDiscrete(grain)
	img.display()


def testVornoUniformPoissonPointProcess3D() :
	domain = Vec3F32(100,100,100)#2d field domain
	lambdafield = 0.0001 #parameter of the Poisson point process
	random = RandomGeometry()

	grain = random.poissonPointProcess(domain,lambdafield)#generate the 2d Poisson point process
	d = DistributionDirac(3)#because the Poisson point process has a volume equal to 0, we associate each point with mono-disperse sphere to display the result
	print lambdafield
	random.sphere(grain,d)
	img = random.continuousToDiscrete(grain)
	print lambdafield
	scene=Scene3d()
	visu = Visualization()
	visu.marchingCube(scene,img)
	visu.lineCube(scene,img)
	print lambdafield
	scene.display()
def testVoronoiTesselation3D() :
	domain = Vec3F32(100,100,100)
	random = RandomGeometry()
	germ = random.poissonPointProcess(domain,0.0001)
	label = random.germToMatrix(germ)
	proc = Processing()
	label = proc.voronoiTesselationEuclidean(label)
	visu = Visualization()
	voronoicolor =  visu.labelToRandomRGB(label)
	scene=Scene3d()
	visu.cube(scene,voronoicolor)
	scene.display()

def testSingleSphere() :
	sphere = GrainSphere3()
	size=100
	sphere.x = Vec3F32(size/2,size/2,size/2)
	sphere.radius = size/2
	germgrain = ModelGermGrain3() 
	germgrain.setDomain(Vec3F32(size,size,size))
	germgrain.grains().push_back(sphere)
	m = RandomGeometry_continuousToDiscrete(germgrain)
	scene = Scene3d() 
	Visualization_marchingCube(scene,m)
	Visualization_lineCube(scene,m)
	scene.display()

def testSingleEllipse() :
	ellipse =  GrainEllipsoid3()
	size=100
	ellipse.x = Vec3F32(size/2,size/2,size/2)
	ellipse.setRadius(Vec3F32(size/2,size/4,size/8))
	
	germgrain = ModelGermGrain3() 
	germgrain.setDomain(Vec3F32(size,size,size))
	germgrain.grains().push_back(ellipse)
	m = RandomGeometry_continuousToDiscrete(germgrain)
	scene = Scene3d() 
	Visualization_marchingCube(scene,m)
	Visualization_lineCube(scene,m)
	scene.display()


def testSingleCylinder() :
	cylinder =  GrainCylinder()
	size=100
	cylinder.x = Vec3F32(size/2,size/2,size/2)
	cylinder.radius = size/8
	cylinder.height = size*3./4.
	angle = OrientationEulerAngle3()
	angle.setAngle_ei(3.14/4.,0)
	angle.setAngle_ei(3.14/4.,1)
	cylinder.orientation = angle
	germgrain = ModelGermGrain3() 
	germgrain.setDomain(Vec3F32(size,size,size))
	germgrain.grains().push_back(cylinder)
	m = RandomGeometry_continuousToDiscrete(germgrain)
	scene = Scene3d() 
	Visualization_marchingCube(scene,m)
	Visualization_lineCube(scene,m)
	scene.display()

def testSingleBox() :
	box =  GrainBox3()
	size=50
	box.x = Vec3F32(size/2,size/2,size/2)
	box.radius = Vec3F32(size/8,size/4,size/2)
	germgrain = ModelGermGrain3() 
	germgrain.setDomain(Vec3F32(size,size,size))
	germgrain.grains().push_back(box)
	m = RandomGeometry_continuousToDiscrete(germgrain)
	scene = Scene3d() 
	Visualization_marchingCube(scene,m)
	Visualization_lineCube(scene,m)
	scene.display()
			
def testSingleRhombohedron() :
	rombohedra =  GrainEquilateralRhombohedron()
	size=100
	rombohedra.x = Vec3F32(size/2,size/2,size/2)
	rombohedra.radius =  size/6
	rombohedra.setAnglePlane(15.*3.14/180.)
	germgrain = ModelGermGrain3() 
	germgrain.setDomain(Vec3F32(size,size,size))
	germgrain.grains().push_back(rombohedra)
	m = RandomGeometry_continuousToDiscrete(germgrain)
	scene = Scene3d() 
	Visualization_marchingCube(scene,m)
	Visualization_lineCube(scene,m)
	scene.display()		


def testSinglePolyhedra() :
	polyhedra = GrainPolyhedra3()
#
# 	#The following Cartesian coordinates define the four vertices of a tetrahedron with edge-length 2, centered at the origin:
	x1 = Vec3F32( 1, 0, -1./math.sqrt(2.))
	x2= Vec3F32(-1, 0, -1./math.sqrt(2.))
	x3= Vec3F32( 0, 1,  1./math.sqrt(2.))
	x4= Vec3F32( 0,-1,  1./math.sqrt(2.))
	
	size=100
	normal_x1_x2_x3 = productVectoriel(x1-x2,x1-x3)
	value = 1
	normal_x1_x2_x3 *=math.copysign(value, productInner(normal_x1_x2_x3,x1))
	normal_x1_x2_x3=normal_x1_x2_x3/normal_x1_x2_x3.norm()
# 	

	polyhedra.addPlane(size/6,normal_x1_x2_x3)
	
	normal_x1_x3_x4 = productVectoriel(x1-x3,x1-x4)
	normal_x1_x3_x4 *=math.copysign(value,productInner(normal_x1_x3_x4,x1));
	normal_x1_x3_x4/=normal_x1_x3_x4.norm()
	polyhedra.addPlane(size/6,normal_x1_x3_x4)
	
	normal_x1_x4_x2 = productVectoriel(x1-x4,x1-x2)
	normal_x1_x4_x2 *=math.copysign(value, productInner(normal_x1_x4_x2,x1))
	normal_x1_x4_x2/=normal_x1_x4_x2.norm()
	polyhedra.addPlane(size/6,normal_x1_x4_x2)

	normal_x2_x3_x4 = productVectoriel(x2-x3,x2-x4);
	normal_x2_x3_x4 *=math.copysign(value, productInner(normal_x2_x3_x4,x2));
	normal_x2_x3_x4/=normal_x2_x3_x4.norm();
	polyhedra.addPlane(size/6,normal_x2_x3_x4);

	polyhedra.x = Vec3F32(size/2,size/2,size/2);
	germgrain = ModelGermGrain3() 
	germgrain.setDomain(Vec3F32(size,size,size))
	germgrain.grains().push_back(polyhedra)
	m = RandomGeometry_continuousToDiscrete(germgrain)
	scene = Scene3d() 
	Visualization_marchingCube(scene,m)
	Visualization_lineCube(scene,m)
	scene.display()		
	
def testProbilityDistributionNormal(porosity = 0.5) :
	d = DistributionNormal(30,5)
	surface_expectation_disk = 3.14*Statistics_moment(d,2,0,60)
	lambda_intensity = -math.log(porosity)/surface_expectation_disk
	domain = Vec2F32(2048,2048)
	germ_grain = RandomGeometry_poissonPointProcess(domain,lambda_intensity)
	RandomGeometry_sphere(germ_grain,d)
	lattice = RandomGeometry_continuousToDiscrete(germ_grain)
	lattice_grey = Mat2UI8(lattice)
	lattice_grey.display()
	volume_fraction = Analysis_histogram(lattice_grey)
	print "expected porosity " + str(porosity)
	print "realization porosity "+str(volume_fraction.getValue(0,1))

def testProbilityDistributionPowerLaw(porosity = 0.5) :
	d = DistributionExpression("1/x^(3.1)")
	minradius = 5
	size = 2048
	d= Statistics_toProbabilityDistribution(d,minradius,size,0.1)
	grain_surface_expectation = 3.14*Statistics_moment(d,2,minradius,size)
	lambda_intensity = -math.log(porosity)/grain_surface_expectation
	domain = Vec2F32(size,size)
	germ_grain = RandomGeometry_poissonPointProcess(domain,lambda_intensity)
	RandomGeometry_sphere(germ_grain,d)
	lattice = RandomGeometry_continuousToDiscrete(germ_grain)
	lattice_grey = Mat2UI8(lattice)
	lattice_grey.display("boolean model")
	volume_fraction = Analysis_histogram(lattice_grey)
	print "expected porosity " + str(porosity)
	print "realization porosity "+str(volume_fraction.getValue(0,1))



#testUniformPoissonPointProcess2D()
#testUniformPoissonPointProcess3D()
#testNonUniformPoissonPointProcess2D()
#testSingleSphere()
#testSingleEllipse()
#testSingleCylinder()
#testSingleBox()
#testSingleRhombohedron()
#testSinglePolyhedra() 
testProbilityDistributionNormal(0.2)
testProbilityDistributionPowerLaw(0.2)
testVoronoiTesselation3D()

