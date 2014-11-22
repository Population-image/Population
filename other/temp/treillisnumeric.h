#ifndef TREILLISNUMERIQUE_H
#define TREILLISNUMERIQUE_H

#include"Population.h"
#include"data/notstable/CharacteristicCluster.h"
//#include"emmintrin.h"
namespace pop
{
class TreillisNumeric
{

public:

    MatNDisplay disp,disp2;

    static bool sortMyFunctionLeft2 (std::pair<int,int>  i,std::pair<int,int> j) ;

    OCRNeuralNetwork ocr;
    Vec<Vec<Mat2UI8> > applyGraphClusterTT(const pop::Mat2UI32& labelled_image, Vec<CharacteristicClusterDistance*> v_dist, Vec<Distribution> v_distribution ,double threshold );
    void treillisNumerique(Mat2UI8 m,int nbr_pixels_width_caracter);
    void treillisNumeriqueVersionText(Mat2UI32 m,int nbr_pixels_width_caracter);
};
}
#endif // TREILLISNUMERIQUE_H
