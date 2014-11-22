#include "ControlEditorPoint.h"
#include<DataPoint.h>
ControlEditorPoint::ControlEditorPoint(QWidget *parent)
    : EditorTable(parent)
{
    this->path().clear();
    this->path().push_back("Point");
    this->setName("EditorPoint");
    this->setKey("EditorPoint");
    this->structurePlug().plugOut().clear();
    this->structurePlug().addPlugOut(DataPoint::KEY,"out.v");
    addRaw();
    buttonraw->hide();rmbuttonraw->hide();
    update();
//    this->setMinimumSize(0,0);
//    this->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
}
//QSize 	ControlEditorPoint::minimumSizeHint () const{
//    return QSize(0,0);
//}

QString ControlEditorPoint::defaultString(){
    return "0";
}

QString ControlEditorPoint::defaultHeaderColumn(int col){
    return "x_"+QString::number(col);
}

QString ControlEditorPoint::defaultHeaderRaw(int ){
    return "";
}
CControl * ControlEditorPoint::clone(){
    return new ControlEditorPoint;
}

void ControlEditorPoint::apply(){
    if(test==true&&this->isPlugOutConnected(0)==true)
    {
        int count = box->columnCount();
        VecF64   t(count);
        for(int i=0;i<count;i++){
            QTableWidgetItem* item = box->item(0,i);
            t(i)=item->text().toFloat();
        }
        DataPoint * table = new DataPoint;
        table->setValue(t);
        this->sendPlugOutControl(0,table,CPlug::NEW);
    }
}

