#ifndef OPERATORVORONOITESSELATIONMatN_H
#define OPERATORVORONOITESSELATIONMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;

class OperatorVoronoiTesselationMatN : public COperator
{
public:
    OperatorVoronoiTesselationMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * incast,BaseMatN * mask, int norm, BaseMatN * &out,BaseMatN * &dist)
        {
            if(MatN<DIM,unsigned char> * maskcast = dynamic_cast<MatN<DIM,unsigned char> *>(mask)){
                //                if(norm!=2){
                MatN<DIM,Type> *outcast =  new MatN<DIM,Type>(incast->getDomain());
                MatN<DIM,pop::UI16> * distcast =  new MatN<DIM,pop::UI16>(incast->getDomain());
                std::pair<MatN<DIM,Type>,MatN<DIM,pop::UI16> > ppair= pop::ProcessingAdvanced::voronoiTesselation(* incast,* maskcast,incast->getIteratorENeighborhood(1,norm));
                *outcast = ppair.first;
                *distcast = ppair.second;
                out=outcast;
                dist = distcast;
                //                }
                //                else{
                //                    MatN<DIM,Type> *outcast =  new MatN<DIM,Type>(incast->getDomain());
                //                    MatN<DIM,pop::F64> * distcastfloat =  new MatN<DIM,pop::F64>(incast->getDomain());

                //                    std::pair<MatN<DIM,Type>,MatN<DIM,pop::F64> > ppair= pop::ProcessingAdvanced::voronoiTesselationEuclidean(* incast,* maskcast);
                //                    *outcast = ppair.first;
                //                    *distcastfloat = ppair.second;

                //                    Img2d_label* distcast =  new Img2d_label(*distcastfloat);

                //                    out=outcast;
                //                    dist = distcast;
                //                }
            }
        }
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * incast, int norm, BaseMatN * &out,BaseMatN * &dist)
        {
            if(norm!=2){
                MatN<DIM,Type> *outcast =  new MatN<DIM,Type>(incast->getDomain());
                MatN<DIM,pop::UI16> * distcast =  new MatN<DIM,pop::UI16>(incast->getDomain());

                std::pair<MatN<DIM,Type>,MatN<DIM,pop::UI16> > ppair= pop::ProcessingAdvanced::voronoiTesselation(* incast,incast->getIteratorENeighborhood(1,norm));
                *outcast = ppair.first;
                *distcast = ppair.second;
                out=outcast;
                dist = distcast;
            }
            else{
                MatN<DIM,Type> *outcast =  new MatN<DIM,Type>(incast->getDomain());
                MatN<DIM,pop::F64> * distcastfloat =  new MatN<DIM,pop::F64>(incast->getDomain());
                std::pair<MatN<DIM,Type>,MatN<DIM,pop::F64> > ppair= pop::ProcessingAdvanced::voronoiTesselationEuclidean(* incast);
                *outcast = ppair.first;
                *distcastfloat = ppair.second;
                MatN<DIM,pop::UI16> * distcast =  new MatN<DIM,pop::UI16>(*distcastfloat);
                out=outcast;
                dist = distcast;
            }
        }
    };

};

#endif // OPERATORVORONOITESSELATIONMatN_H
