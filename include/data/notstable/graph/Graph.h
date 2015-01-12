/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has resulted from the use of the
software population, the reference of this book "Population library, 2012,
Vincent Tariel" shall be included in it.

So, the terms of the Population License are:

Copyright © 2012-2015, Tariel Vincent

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software and for any writing
public or private that has resulted from the use of the software population,
the reference of this book "Population library, 2012, Vincent Tariel" shall
be included in it.

The Software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising
from, out of or in connection with the software or the use or other dealings
in the Software.
\***************************************************************************/
#ifndef Graph_H
#define Graph_H
#include<vector>
#include<iostream>
#include<fstream>
#include<string>
#include"PopulationConfig.h"
#include"data/utility/BasicUtility.h"
namespace pop
{

class GraphIteratorEDomain
{
private:
    int _nbr_vertex;
    int _current_vertex;
public:
    typedef  int Domain;
    GraphIteratorEDomain(const GraphIteratorEDomain& it)
        :_nbr_vertex(it.getDomain())
    {
        init();
    }

    GraphIteratorEDomain(const Domain & domain)
        :    _nbr_vertex(domain){init();}
    void init()
    {
        _current_vertex =-1;
    }
    bool next()
    {
        _current_vertex++;
        if(_current_vertex<_nbr_vertex){
            return true;
        }
        else{
            _current_vertex=0;
            return false;
        }
    }
    int & x()
    {
        return _current_vertex;
    }
    int getDomain()const
    {
        return _nbr_vertex;
    }
};
template<typename AdjencyList,typename EdgeLink>
class GraphIteratorENeighborhood
{
protected:
    int _label_vertex;
    int _index;
    bool _withcenter;
    const AdjencyList * _adgency;
    const EdgeLink * _edge_link;
public:
    typedef std::pair<const AdjencyList *,const EdgeLink *> Domain;
    GraphIteratorENeighborhood( Domain domain)
        :  _label_vertex(0),_withcenter(true),_adgency(domain.first),_edge_link(domain.second){}

    Domain getDomain()const{return std::make_pair(_adgency,_edge_link);}

    void removeCenter(){_withcenter = false;}
    void init(int label_vertex)
    {
        _label_vertex = label_vertex;
        if(_withcenter==true)
            _index=-2;
        else
            _index=-1;

    }
    bool next()
    {
        _index++;
        if(_index==-1)
            return true;
        else if(_index<static_cast<int>((*_adgency)[_label_vertex].size()))
            return true;
        else
            return false;
    }
    int x()
    {
        if(_index==-1)
            return _label_vertex;
        if( (*_edge_link)[(*_adgency)[_label_vertex][_index] ].first ==_label_vertex )
            return (*_edge_link)[(*_adgency)[_label_vertex][_index] ].second;
        else
            return (*_edge_link)[(*_adgency)[_label_vertex][_index] ].first;
    }
};

template<typename VertexType=NullType,typename EdgeType=NullType>
class POP_EXPORTS GraphAdjencyList
{
public:
    enum{
        NO_EDGE_INDEX=-1
    };
    std::vector<VertexType> _v_vertex;
    std::vector<std::vector<int> > _v_adjency_list;
    std::vector<EdgeType> _v_edge;
    std::vector<std::pair<int,int> > _v_edge_link;

    typedef int E;
    typedef VertexType F;
    typedef GraphAdjencyList< VertexType, EdgeType> Domain;
    typedef GraphIteratorEDomain IteratorEDomain;
    typedef GraphIteratorENeighborhood<std::vector<std::vector<int> >, std::vector<std::pair<int,int> > > IteratorENeighborhood;
    GraphAdjencyList(){

    }
    GraphAdjencyList(unsigned int nbr_vertex, VertexType v)
        :_v_vertex(nbr_vertex,v),_v_adjency_list(nbr_vertex){}

    GraphAdjencyList(const GraphAdjencyList & graph, VertexType v)
    {
        _v_vertex.resize(graph._v_vertex.size());
        std::fill(_v_vertex.begin(),_v_vertex.end(),v);
        _v_adjency_list.resize(graph._v_adjency_list.size());
        std::copy(graph._v_adjency_list.begin(),graph._v_adjency_list.end(),_v_adjency_list.begin());
        _v_edge.resize(graph._v_edge.size());
        std::copy(graph._v_edge.begin(),graph._v_edge.end(),_v_edge.begin());
        _v_edge_link.resize(graph._v_edge_link.size());
        std::copy(graph._v_edge_link.begin(),graph._v_edge_link.end(),_v_edge_link.begin());
    }
    template<typename VertexType1,typename EdgeType1>
    GraphAdjencyList(const GraphAdjencyList< VertexType1, EdgeType1> graph){
        _v_vertex.resize(graph._v_vertex.size());
        std::copy(graph._v_vertex.begin(),graph._v_vertex.end(),_v_vertex.begin());
        _v_adjency_list.resize(graph._v_adjency_list.size());
        std::copy(graph._v_adjency_list.begin(),graph._v_adjency_list.end(),_v_adjency_list.begin());
        _v_edge.resize(graph._v_edge.size());
        std::copy(graph._v_edge.begin(),graph._v_edge.end(),_v_edge.begin());
        _v_edge_link.resize(graph._v_edge_link.size());
        std::copy(graph._v_edge_link.begin(),graph._v_edge_link.end(),_v_edge_link.begin());
    }

    IteratorEDomain getIteratorEDomain()const{
        return IteratorEDomain(static_cast<int>(_v_vertex.size()));
    }
    IteratorENeighborhood getIteratorENeighborhood(int =1,int =-1)const{
        return IteratorENeighborhood(std::make_pair(&_v_adjency_list,&_v_edge_link));
    }

    GraphAdjencyList< VertexType, EdgeType> getDomain()const{
        return *this;
    }

    int addVertex();
    VertexType & vertex(int vertex_label);
    const VertexType & vertex(int vertex_label)const;
    unsigned int sizeVertex()const;

    VertexType & operator()(int vertex_label){
        return _v_vertex[vertex_label];
    }
    const VertexType & operator()(int vertex_label)const{
        return _v_vertex[vertex_label];
    }


    int addEdge();
    EdgeType & edge(int edge_label);
    const EdgeType & edge(int edge_label)const;
    unsigned int sizeEdge()const;
    void connection(int edge_label, int vertex_label1,int vertex_label2);

    const std::vector<std::pair<int,int> >& links()const;
    std::pair<int,int> getLink(int edge_label)const;

    int getEdge(int vertex_label1,int vertex_label2)const;
    const std::vector<int>& getEdges(int vertex_label1)const;
    std::vector<int> getConnectedVertex(int vertex_label)const;


        void load(std::string file);
        void save(std::string file)const;
};
template<typename VertexType1,typename EdgeType,typename VertexType2>
struct FunctionTypeTraitsSubstituteF<GraphAdjencyList<VertexType1,EdgeType>,VertexType2 >
{
    typedef GraphAdjencyList<VertexType2,EdgeType> Result;
};

template<typename VertexType,typename EdgeType>
std::pair<int,int> GraphAdjencyList<VertexType,EdgeType>::getLink(int edge_label) const{
    return _v_edge_link[edge_label];
}
template<typename VertexType,typename EdgeType>
const std::vector<std::pair<int,int> >& GraphAdjencyList<VertexType,EdgeType>::links()const{
    return _v_edge_link;
}

template<typename VertexType,typename EdgeType>
int GraphAdjencyList<VertexType,EdgeType>::addVertex(){
    _v_vertex.push_back(VertexType());
    _v_adjency_list.push_back( std::vector<int>());
    return  (int)_v_vertex.size()-1;

}
template<typename VertexType,typename EdgeType>
unsigned int GraphAdjencyList<VertexType,EdgeType>::sizeVertex()const{
    return (int)_v_vertex.size();
}
template<typename VertexType,typename EdgeType>
unsigned int GraphAdjencyList<VertexType,EdgeType>::sizeEdge() const{
    return (int)_v_edge.size();
}
template<typename VertexType,typename EdgeType>
VertexType & GraphAdjencyList<VertexType,EdgeType>::vertex(int vertex_label){
    return _v_vertex[vertex_label];
}
template<typename VertexType,typename EdgeType>
const VertexType & GraphAdjencyList<VertexType,EdgeType>::vertex(int vertex_label)const{
    return _v_vertex[vertex_label];
}
template<typename VertexType,typename EdgeType>
int GraphAdjencyList<VertexType,EdgeType>::addEdge(){
    _v_edge.push_back(EdgeType());
    _v_edge_link.push_back(std::make_pair(-1,-1));
    return  (int)_v_edge.size()-1;
}
template<typename VertexType,typename EdgeType>
EdgeType & GraphAdjencyList<VertexType,EdgeType>::edge(int edge_label){
    return _v_edge[edge_label];
}
template<typename VertexType,typename EdgeType>
const EdgeType & GraphAdjencyList<VertexType,EdgeType>::edge(int edge_label)const{
    return _v_edge[edge_label];
}
template<typename VertexType,typename EdgeType>
void GraphAdjencyList<VertexType,EdgeType>::connection(int edge_label, int vertex_label1,int vertex_label2){
    _v_adjency_list[vertex_label1].push_back(edge_label);
    _v_adjency_list[vertex_label2].push_back(edge_label);
    _v_edge_link[edge_label] = std::make_pair(vertex_label1,vertex_label2);
}
template<typename VertexType,typename EdgeType>
int GraphAdjencyList<VertexType,EdgeType>::getEdge(int vertex_label1,int vertex_label2) const{
    std::vector<int>::const_iterator it;
    for ( it=_v_adjency_list[vertex_label1].begin() ; it < _v_adjency_list[vertex_label1].end(); it++ ){
        int edge_label = *it;
        if(_v_edge_link[edge_label].second ==vertex_label2 )
            return edge_label;
        else if(_v_edge_link[edge_label].first ==vertex_label2)
            return edge_label;
    }
    return NO_EDGE_INDEX;
}
template<typename VertexType,typename EdgeType>
const std::vector<int> &GraphAdjencyList<VertexType,EdgeType>::getEdges(int vertex_label) const{
    return _v_adjency_list[vertex_label] ;
}
template<typename VertexType,typename EdgeType>
std::vector<int> GraphAdjencyList<VertexType,EdgeType>::getConnectedVertex(int vertex_label) const{
    std::vector<int> v_vertex;
    std::vector<int>::const_iterator it;
    for ( it=_v_adjency_list[vertex_label].begin() ; it < _v_adjency_list[vertex_label].end(); it++ ){
        int edge_label = *it;
        if(_v_edge_link[edge_label].second !=vertex_label )
            v_vertex.push_back(_v_edge_link[edge_label].second);
        else
            v_vertex.push_back(_v_edge_link[edge_label].first);
    }
    return v_vertex;
}



template<typename VertexType,typename EdgeType>
std::ostream& operator << (std::ostream& out, const GraphAdjencyList<VertexType,EdgeType>& m){
    out<<"#NBR_VECTOR"<<std::endl;
    out<<m._v_vertex.size()<<std::endl;
    out<<"#DATA_VECTOR"<<std::endl;
    for ( int i =0; i < (int)m._v_vertex.size(); i++ ){
        out<<m._v_vertex[i]<<std::endl;
    }
    out<<"#NBR_EDGE"<<std::endl;
    out<<m._v_edge.size()<<std::endl;
    out<<"#DATA_EDGE"<<std::endl;
    for ( int i =0; i < (int)m._v_edge.size(); i++ ){
        out<<m._v_edge[i]<<std::endl;
        out<<m._v_edge_link[i].first<<"\t"<<m._v_edge_link[i].second<<std::endl;
    }
    return out;
}
template<typename VertexType,typename EdgeType>
std::istream& operator >> (std::istream& in, GraphAdjencyList<VertexType,EdgeType>& m){
    std::string str;
    in >> str;
    in >> str;
    int nbrvertex;
    in>>nbrvertex;
    in >> str;
    for(int i =0;i<nbrvertex;i++)
    {
        m.addVertex();
        VertexType v;
        in >> v;
        m.vertex(i)=v;
    }
    in >> str;
    int nbredge;
    in>>nbredge;
    in >> str;
    for(int i =0;i<nbredge;i++)
    {
        m.addEdge();
        EdgeType e;
        in>>e;
        m.edge(i)=e;
        int v1,v2;
        in>>v1;
        in>>v2;
        m.connection(i,v1,v2);

    }
    return in;
}

template<typename VertexType,typename EdgeType>
void GraphAdjencyList<VertexType,EdgeType>::load(std::string file){
    std::ifstream  in(file.c_str());
    if (in.fail())
    {
        std::cout<<"GraphAdjencyList: cannot open file: "<<file<<std::endl;
    }
    else
    {
        in>>*this;
    }

}
template<typename VertexType,typename EdgeType>
void GraphAdjencyList<VertexType,EdgeType>::save(std::string file) const{
    std::ofstream  out(file.c_str());
    if (out.fail())
    {
        std::cout<<"GraphAdjencyList: cannot open file: "<<file<<std::endl;
    }
    else
    {

        out<<*this;
    }
}

}
#endif // Graph_H
