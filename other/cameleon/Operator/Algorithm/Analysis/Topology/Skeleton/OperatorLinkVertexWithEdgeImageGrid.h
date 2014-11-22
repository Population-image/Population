#ifndef OPERATORLINKVERTEXWITHEDGEMatN_H
#define OPERATORLINKVERTEXWITHEDGEMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;

class OperatorLinkEdgeVertexMatN : public COperator
{
public:
    OperatorLinkEdgeVertexMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM, typename TYPE>
        void operator()(MatN<DIM,TYPE> * vertex,BaseMatN* edgenocast, GraphBase * &g,int & tore)
        {

            MatN<DIM,TYPE> * edge=dynamic_cast<MatN<DIM,TYPE> *>(edgenocast);

            GraphAdjencyList<VertexPosition,Edge> * gcast = new GraphAdjencyList<VertexPosition,Edge>;
            (*gcast) = Analysis::linkEdgeVertex(  *vertex, *edge,tore);
            g = gcast;
        }
    };

};

#endif // OPERATORLINKVERTEXWITHEDGEMatN_H
