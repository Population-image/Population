#include"Population.h"//Single header
using namespace pop;//Population namespace

int main()
{
	while(1==1){
		    omp_set_num_threads(1);
		pop::Mat2UI8 m(6000,6000);
	
		
		int time1 = time(NULL);
		for(unsigned int i=0;i<1;i++){
			m=Processing::thresholdNiblackMethod(m);
		}
	//	m = m*m;
	int time2 = time(NULL);
	std::cout<<time2-time1<<std::endl;
	}
	return 0;
    Mat2UI8 img;//2d grey-level image object
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
    int value;
    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
    threshold.save("iexthreshold.pgm");
    Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
    color.display();
    return 0;
}
