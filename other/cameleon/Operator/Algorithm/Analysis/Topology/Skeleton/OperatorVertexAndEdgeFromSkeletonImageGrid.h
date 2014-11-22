#ifndef OPERATORVERTEXANDEDGEFROMSKELETONMatN_H
#define OPERATORVERTEXANDEDGEFROMSKELETONMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorVertexAndEdgeFromSkeletonMatN : public COperator
{
public:
    OperatorVertexAndEdgeFromSkeletonMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,BaseMatN * &vertex,BaseMatN * &edge)
        {
            MatN<DIM,unsigned char> *vertexcast = new MatN<DIM,unsigned char> (in1cast->getDomain());
            MatN<DIM,unsigned char> *edgecast =   new MatN<DIM,unsigned char> (in1cast->getDomain());
            pair< MatN<DIM,unsigned char>,MatN<DIM,unsigned char> >ppair = Analysis::fromSkeletonToVertexAndEdge(*in1cast);
            *vertexcast = ppair.first;
            *edgecast = ppair.second;
            vertex = vertexcast;
            edge = edgecast;
        }
    };

};

#endif // OPERATORVERTEXANDEDGEFROMSKELETONMatN_H
