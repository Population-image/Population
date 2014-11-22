#ifndef OPERATORNONLINEARANISTROPICDIFFUSIONMatN_H
#define OPERATORNONLINEARANISTROPICDIFFUSIONMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/PDE.h"
using namespace pop;
class OperatorNonLinearAnistropicDiffusionMatN : public COperator
{
public:
    OperatorNonLinearAnistropicDiffusionMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * init,int nbrsteps,double kappa, double sigma, BaseMatN * &end){
            MatN<DIM,Type> * endcast = new MatN<DIM,Type>(init->getDomain());
            * endcast = PDE::nonLinearAnisotropicDiffusionDericheFast( * init,nbrsteps,kappa,sigma);
            end = endcast;
        }
    };
        void initState();

};

#endif // OPERATORNONLINEARANISTROPICDIFFUSIONMatN_H
