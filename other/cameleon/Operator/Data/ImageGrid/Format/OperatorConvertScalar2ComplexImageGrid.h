#ifndef OPERATORCONVERTSCALAR2COMPLEXMatN_H
#define OPERATORCONVERTSCALAR2COMPLEXMatN_H


#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Visualization.h"
using namespace pop;

class OperatorConvertScalar2ComplexMatN : public COperator
{
public:
    OperatorConvertScalar2ComplexMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type > * realcast, BaseMatN * img, BaseMatN * &complex)throw(pexception)
        {

            MatN<DIM,ComplexF64 > * complexcast = new MatN<DIM,ComplexF64 >(realcast->getDomain());
            if(MatN<DIM,Type > * imgcast = dynamic_cast<MatN<DIM,Type>* >(img) )
            {
                Convertor::fromRealImaginary(*realcast,*imgcast,*complexcast);
            }
            else
            {
                throw(std::string("pixel/voxel type real = pixel/voxel type img"));
            }
            complex=complexcast;
        }
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type > * realcast, BaseMatN * &complex)throw(pexception)
        {

            MatN<DIM,ComplexF64 > * complexcast = new MatN<DIM,ComplexF64 >(realcast->getDomain());
            Convertor::fromRealImaginary(*realcast,*complexcast);
            complex=complexcast;
        }
    };

};
#endif // OPERATORCONVERTSCALAR2COMPLEXMatN_H
