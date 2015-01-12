#include"Population.h"//Single header
using namespace pop;//Population namespace
int main(){
    NeuralNetworkFeedForward n;
    TrainingNeuralNetwork::neuralNetworkForRecognitionForHandwrittenDigits(n,"/home/vincent/train-images.idx3-ubyte",
                                                                           "/home/vincent/train-labels.idx1-ubyte",
                                                                           "/home/vincent/t10k-images.idx3-ubyte",
                                                                           "/home/vincent/t10k-labels.idx1-ubyte",1,0);
    Mat2UI8 img;
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
    img.display("Initial image",false);
    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
    img.display();
    double value;
    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
    threshold.save("iexthreshold.png");
    Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
    color.display("Segmented image",true);
    return 0;
}
