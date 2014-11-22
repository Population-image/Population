import Image
import population   

def toPIL(imgpop):
	## RGB image ###
	if type(imgpop) == type(population.Mat2RGBUI8()):
		dataseq = population.ArrayRGBUI8.frompointer(m.data())#acces the data with an array
		img=Image.new('RGB' ,(m.sizeJ(),m.sizeI()))
		pixels = img.load()
		for i in range(0,img.size[0]):
			for j in range(0,img.size[1]):
				rgb = dataseq[j*m.sizeJ()+i]
				pixels[i,j] = (rgb.getValue(0),rgb.getValue(1),rgb.getValue(2)) 
		return img
	## grey image in 8 bytes ###
	elif type(imgpop) == type(population.Mat2UI8()):
		dataseq = population.ArrayUI8.frompointer(m.data())#acces the data with an array
		img=Image.new('L' ,(m.sizeJ(),m.sizeI()))
		pixels = img.load()
		for i in range(0,img.size[0]):
			for j in range(0,img.size[1]):
				pixels[i,j] = dataseq[j*m.sizeJ()+i]
		return img
	else :
		return Image.new('RGB' ,(m.sizeJ(),m.sizeI()))

def fromPIL(imgpil):
	## RGB image ###
	if(  imgpil.mode=="RGB"):
		dataseq =population.ArrayRGBUI8(imgpil.size[0]*imgpil.size[1])#create an array
		pixels = imgpil.load()
		for i in range(0,imgpil.size[0]):
			for j in range(0,imgpil.size[1]):
		     		dataseq[j*m.sizeJ()+i]= population.RGBUI8(pixels[i,j][0],pixels[i,j][1],pixels[i,j][2])	
		pixelvalue = population.ArrayRGBUI8.cast(dataseq)#cast the array to the Swig type
		imgpop = population.Mat2RGBUI8(population.Vec2I32(img.size[1],img.size[0]),pixelvalue)#set 
		return imgpop
	## grey image in 8 bytes ###
	if(  imgpil.mode=="L"):
		dataseq =population.ArrayUI8(imgpil.size[0]*imgpil.size[1])#create an array
		pixels = imgpil.load()
		for i in range(0,imgpil.size[0]):
			for j in range(0,imgpil.size[1]):
		     		dataseq[j*m.sizeJ()+i]= pixels[i,j]	
		pixelvalue = population.ArrayUI8.cast(dataseq)#cast the array to the Swig type
		imgpop = population.Mat2UI8(population.Vec2I32(img.size[1],img.size[0]),pixelvalue)#set it
		return imgpop

m = population.Mat2UI8()
m.load("lena.pgm")
img = toPIL(m)
imgpop =fromPIL(img)
imgpop.display()
m = population.Mat2RGBUI8()
m.load("lena.jpg")
img = toPIL(m)
imgpop =fromPIL(img)
imgpop.display()

