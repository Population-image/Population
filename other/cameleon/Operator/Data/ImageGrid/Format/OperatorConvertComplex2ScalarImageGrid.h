#ifndef OPERATORCONVERTCOMPLEX2SCALARMatN_H
#define OPERATORCONVERTCOMPLEX2SCALARMatN_H


#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Visualization.h"
using namespace pop;

class OperatorConvertComplex2ScalarMatN : public COperator
{
public:
    OperatorConvertComplex2ScalarMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM>
        void operator()(MatN<DIM,ComplexF64 > * complexcast, BaseMatN * &real, BaseMatN * &img){
              MatN<DIM,pop::F64> * realcast = new MatN<DIM,pop::F64>(complexcast->getDomain());
              MatN<DIM,pop::F64> * imgcast = new MatN<DIM,pop::F64>(complexcast->getDomain());
              Convertor::toRealImaginary(* complexcast,*realcast,*imgcast);
              real=realcast;
              img=imgcast;
        }
    };

};
#endif // OPERATORCONVERTCOMPLEX2SCALARMatN_H
