#ifndef DATAOPENGL_H
#define DATAOPENGL_H

#include<CDataByFile.h>
#include"data/3d/GLFigure.h"
using namespace pop;
class DataOpenGl : public CDataByFile<Scene3d>
{
public:
    DataOpenGl();
     static string KEY;
     DataOpenGl * clone();
     shared_ptr<Scene3d> getDataByFile();
     void setDataByFile(shared_ptr<Scene3d> type);
     void setDataByCopy(shared_ptr<Scene3d> type);
};
#endif // DATAOPENGL_H
