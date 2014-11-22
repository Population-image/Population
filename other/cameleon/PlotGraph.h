/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has resulted from the use of the
software population, the reference of this book "Population library, 2012,
Vincent Tariel" shall be included in it.

So, the terms of the Population License are:

Copyright © 2012, Tariel Vincent

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

#ifndef PLOTGRAPH_H
#define PLOTGRAPH_H
#include<queue>
#include<string>
#include"data/typeF/TypeF.h"
#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"

namespace pop
{
class POP_EXPORTS PlotSingleGraph
{

public:
    enum TypeGraph{
        LINE,
        DOT
    };
    PlotSingleGraph();
    PlotSingleGraph(const PlotSingleGraph & plot);
    void setPoints(const std::vector<F64> &x,const std::vector<F64> &y);
    void setLegend(std::string name);
    void setRGB(const RGBUI8 & RGB);
    void setWidth(F64 width);
    void setBrush(const RGBUI8 & RGB);
    void setAlpha(F64 alpha);

    void setTypeGraph(TypeGraph type);

    const std::deque<F64>  & X()const;
    const std::deque<F64>  & Y()const;

    std::deque<F64>  & X();
    std::deque<F64>  & Y();
     std::string getLegend()const;
    RGBUI8 getRGB()const;
    F64 getWidth()const;
    RGBUI8 getBrush()const;
    F64 getAlpha()const;

    TypeGraph getTypeGraph()const;

    void fromMatrix(const Mat2F64 & matrix, unsigned int col_for_xaxis,unsigned int col_for_yaxis)throw(pexception);
private:
    std::deque<F64> _x;
    std::deque<F64> _y;
     std::string _name;
    RGBUI8 _RGB;
    RGBUI8 _brush;
    F64 _alpha;
    TypeGraph _type;
    F64 _width;
};

class POP_EXPORTS PlotGraph
{
public:

    PlotGraph();
    PlotGraph(const PlotSingleGraph & plot);
    PlotGraph(const PlotGraph & plot);
//    PlotGraph & operator =(const PlotGraph & plot);
    void setXAxixLegend(std::string name);
    void setYAxixLegend(std::string name);
     std::string getXAxisLegend()const;
     std::string getYAxisLegend()const;


    void setXAxisLog(bool xaxislog);
    void setYAxisLog(bool yaxislog);

    bool getXAxisLog()const;
    bool getYAxisLog()const;


    void setTitle(std::string title);
     std::string getTitle()const;
    std::vector<PlotSingleGraph> & VGraph();
    const std::vector<PlotSingleGraph> & VGraph()const;
    void save(std::string file);
    void load(std::string file);
private:
    bool _logxaxis;
    bool _logyaxis;
     std::string _xaxisname;
     std::string _yaxisname;
     std::string _title;
    std::vector<PlotSingleGraph> _v_graph;


};
PlotSingleGraph PlotGraphProcedureFromMatrix(const pop::Mat2F64 & m,unsigned int col_for_xaxis,unsigned int col_for_yaxis)throw(pop::pexception);
}
#endif // PLOTGRAPH_H
