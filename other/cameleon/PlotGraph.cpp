#include"data/utility/PlotGraph.h"
namespace pop{
PlotSingleGraph::PlotSingleGraph()
    :_RGB(0,0,0),_brush(255,255,255),_alpha(0),_type(LINE),_width(1)
{

}
PlotSingleGraph::PlotSingleGraph(const PlotSingleGraph & plot){
    _x = plot.X();
    _y = plot.Y();
    _name= plot.getLegend();
    _RGB= plot.getRGB();
    _type= plot.getTypeGraph();
    _width=plot.getWidth();
    _brush= plot.getBrush();
    _alpha=plot.getAlpha();
}

void PlotSingleGraph::fromMatrix(const Mat2F64 & m,unsigned int col_for_xaxis,unsigned int col_for_yaxis)throw(pexception)
{
    if(col_for_xaxis<m.sizeJ() &&  col_for_yaxis<m.sizeJ()){
        std::vector<F64> x(m.sizeI()), y(m.sizeI()); // initialize with 101 entries
        for(unsigned int i=0;i<m.sizeI();i++){
            x[i] =m(i,col_for_xaxis);
            y[i] =m(i,col_for_yaxis);
        }
        this->setPoints(x,y);
    }else{
        throw(pexception("In PlotSingleGraph::fromMatrix, out of range of xaxis or yaxis :PlotGraphProcedureFromMatrix"));
    }
}


void PlotSingleGraph::setPoints(const std::vector<F64> &x,const std::vector<F64> &y){
    _x.clear();
    _y.clear();
    for(int i =0;i<(int)x.size();i++){
        _x.push_back(x[i]);
        _y.push_back(y[i]);

    }


}
void PlotSingleGraph::setLegend(std::string name){
    _name = name;
}

void PlotSingleGraph::setRGB(const RGBUI8 & RGB){
    _RGB= RGB;
}

void PlotSingleGraph::setWidth(F64 width){
    _width = width;
}
void PlotSingleGraph::setBrush(const RGBUI8 & RGB){
    _brush= RGB;
}

void PlotSingleGraph::setAlpha(F64 alpha){
    _alpha = alpha;
}
void PlotSingleGraph::setTypeGraph(TypeGraph type){
    _type = type;
}
const std::deque<F64>&  PlotSingleGraph::X()const{
    return _x;
}

const std::deque<F64>& PlotSingleGraph::Y()const{
    return _y;
}
 std::deque<F64>&  PlotSingleGraph::X(){
    return _x;
}

 std::deque<F64>& PlotSingleGraph::Y(){
    return _y;
}

std::string PlotSingleGraph::getLegend()const{
    return _name;
}

RGBUI8 PlotSingleGraph::getRGB()const{
    return _RGB;
}

F64 PlotSingleGraph::getWidth() const{
    return _width;
}
RGBUI8 PlotSingleGraph::getBrush()const{
    return _brush;
}

F64 PlotSingleGraph::getAlpha() const{
    return _alpha;
}
PlotSingleGraph::TypeGraph PlotSingleGraph::getTypeGraph()const{
    return _type;
}
void PlotGraph::setXAxixLegend(std::string name){
    _xaxisname = name;
}

void PlotGraph::setYAxixLegend(std::string name){
    _yaxisname = name;
}

std::string PlotGraph::getXAxisLegend()const{
    return _xaxisname;
}

std::string PlotGraph::getYAxisLegend()const{
    return _yaxisname;
}

void PlotGraph::setTitle(std::string title){
    _title = title;
}

std::string PlotGraph::getTitle()const{
    return _title;
}
void PlotGraph::setXAxisLog(bool xaxislog){
    _logxaxis = xaxislog;
}

void PlotGraph::setYAxisLog(bool yaxislog){
    _logyaxis = yaxislog;
}

bool PlotGraph::getXAxisLog()const{
    return _logxaxis;
}
bool PlotGraph::getYAxisLog()const{
    return _logyaxis;
}
std::vector<PlotSingleGraph> & PlotGraph::VGraph(){
    return _v_graph;
}
const std::vector<PlotSingleGraph> & PlotGraph::VGraph()const{
    return _v_graph;
}
void PlotGraph::save(std::string ){
    //TODO
}

void PlotGraph::load(std::string ){
    //TODO

}
PlotGraph::PlotGraph(const PlotGraph & plot){
    _xaxisname=plot.getXAxisLegend();
    _yaxisname=plot.getYAxisLegend();
    _title=plot.getTitle();
    _v_graph=plot.VGraph();
    _logxaxis = plot.getXAxisLog();
    _logyaxis = plot.getYAxisLog();
}
//PlotGraph & PlotGraph::operator =(const PlotGraph & plot){
//    _xaxisname=plot.getXAxisLegend();
//    _yaxisname=plot.getYAxisLegend();
//    _title=plot.getTitle();
//    _v_graph=plot.VGraph();
//    return *this;
//}

PlotGraph::PlotGraph(){
    _xaxisname="x";
    _yaxisname="y";
    _logxaxis =false;
    _logyaxis =false;
}

PlotGraph::PlotGraph(const PlotSingleGraph & plot){
    _xaxisname="x";
    _yaxisname="y";
    _v_graph.push_back(plot);
    _logxaxis =false;
    _logyaxis =false;
}
PlotSingleGraph PlotGraphProcedureFromMatrix(const pop::Mat2F64 & m,unsigned int col_for_xaxis,unsigned int col_for_yaxis)throw(pop::pexception)
{
    if(col_for_xaxis<m.sizeJ() && col_for_yaxis<m.sizeJ()){
        std::vector<F64> x(m.sizeI()), y(m.sizeI()); // initialize with 101 entries
        for(unsigned int i=0;i<m.sizeI();i++){
            x[i] =m(i,col_for_xaxis);
            y[i] =m(i,col_for_yaxis);
        }
        PlotSingleGraph c;
        c.setPoints(x,y);

        return c;
    }else{
        throw(pexception("Out of range of xaxis or yaxis :PlotGraphProcedureFromMatrix"));
    }
}
}
