#ifndef OPERATORSETBALLMatN_H
#define OPERATORSETBALLMatN_H
#include"COperator.h"
#include"data/mat/MatN.h"
using namespace pop;
class OperatorSetBallMatN : public COperator
{
public:

    OperatorSetBallMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,RGB<Type> > * in1cast, VecF64  x,double r, double g,  double b,double radius, double norm)
        {
            VecN<DIM,pop::F64> p;
            p=x;


            typename MatN<DIM,RGB<Type> >::IteratorENeighborhood it(in1cast->getIteratorENeighborhood(radius,norm));
            it.init(p);
            while(it.next()){
                in1cast->operator ()(it.x()).r()=r;
                in1cast->operator ()(it.x()).g()=g;
                in1cast->operator ()(it.x()).b()=b;
            }
        }
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast, VecF64  x,double scalar,double radius, double norm)
        {
            VecN<DIM,pop::F64> p;
            p=x;
            typename MatN<DIM,RGBUI8 >::IteratorENeighborhood it(in1cast->getIteratorENeighborhood(radius,norm));
            it.init(p);
            while(it.next()){
                in1cast->operator ()(it.x()) =  scalar;
            }
        }
    };
};

#endif // OPERATORSETBALLMatN_H
