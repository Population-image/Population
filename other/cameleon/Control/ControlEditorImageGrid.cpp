#include "ControlEditorImageGrid.h"
#include "CMachine.h"

#include"data/mat/MatN.h"

#include"data/mat/MatNInOut.h"

QString ControlEditorMatN2D::defaultHeaderColumn(int col){
    return "X"+QString::number(col);
}

QString ControlEditorMatN2D::defaultHeaderRaw(int raw){
    return "Y"+QString::number(raw);
}

ControlEditorMatN2D::ControlEditorMatN2D(QWidget *parent)
    : ControlEditorMatrix(parent)
{
    this->path().clear();
    this->path().push_back("ImageGrid");
    this->setName("EditorMatN2D");
    this->setKey("ControlEditorMatN2D");
    this->setInformation("Create a 2D MatN with float voxel type");
    this->structurePlug().plugOut().clear();
    this->structurePlug().addPlugOut(DataMatN::KEY,"out.pgm");
    buttoncolumn->setText("addX");
    buttonraw->setText("addY");
    rmbuttoncolumn->setText("rmX");
    rmbuttonraw->setText("rmY");
}

CControl * ControlEditorMatN2D::clone(){
    return new ControlEditorMatN2D;
}

void ControlEditorMatN2D::apply(){
    if(test==true&&this->isPlugOutConnected(0)==true)
    {
        VecN<2,int> d;
        d(1) = box->rowCount();
        d(0) = box->columnCount();
        MatN<2,pop::F64>* t = new MatN<2,pop::F64>(d);
        for(int i=0;i<box->rowCount();i++){
            for(int j=0;j<box->columnCount();j++){
                QTableWidgetItem* item = box->item(i,j);
                d(1) = i;
                    d(0) = j;
                t->operator ()(d)=item->text().toDouble();
            }
        }
        DataMatN * data = new DataMatN;
        shared_ptr<BaseMatN> st(t);
        data->setData(st);
        this->sendPlugOutControl(0,data,CPlug::NEW);
    }
}


ControlEditorMatN3D::ControlEditorMatN3D(QWidget * parent)
    :CControl(parent)
{
    this->path().push_back("ImageGrid");
    this->setName("EditorMatN3D");
    this->setKey("ControlEditorImageGrid");
    this->structurePlug().addPlugOut(DataMatN::KEY,"out.pgm");
    this->setInformation("Create a 3D MatN with float voxel type");

    box = new QTableWidget(this);
    box->setRowCount(0);
    box->setColumnCount(0);
    addXTable();addXTable();addXTable();
    addYTable();addYTable();addYTable();

    VecN<3,int> d =3;
    img.resize(d);
    _slice = 0;



    spinz = new QSpinBox;
    spinz->setMinimum(0);
    spinz->setMaximum(2);
    spinz->setValue(0);
    buttonaddx = new QPushButton("addX");
    buttonaddy = new QPushButton("addY");
    buttonaddz = new QPushButton("addZ");
    buttonrmx = new QPushButton("rmX");
    buttonrmy = new QPushButton("rmY");
    buttonrmz = new QPushButton("rmZ");


    save = new QPushButton("save");

    if(!QObject::connect(buttonaddx, SIGNAL(clicked()),this, SLOT(addX()),Qt::DirectConnection)){
        //qDebug << "[WARN] Can't connect EditorTable and box" ;
    }
    if(!QObject::connect(buttonaddy, SIGNAL(clicked()),this, SLOT(addY()),Qt::DirectConnection)){
        //qDebug << "[WARN] Can't connect EditorTable and box" ;
    }

    if(!QObject::connect(buttonaddz, SIGNAL(clicked()),this, SLOT(addZ()),Qt::DirectConnection)){
        //qDebug << "[WARN] Can't connect EditorTable and box" ;
    }
    if(!QObject::connect(buttonrmx, SIGNAL(clicked()),this, SLOT(rmX()),Qt::DirectConnection)){
        //qDebug << "[WARN] Can't connect EditorTable and box" ;
    }
    if(!QObject::connect(buttonrmy, SIGNAL(clicked()),this, SLOT(rmY()),Qt::DirectConnection)){
        //qDebug << "[WARN] Can't connect EditorTable and box" ;
    }

    if(!QObject::connect(buttonrmz, SIGNAL(clicked()),this, SLOT(rmZ()),Qt::DirectConnection)){
        //qDebug << "[WARN] Can't connect EditorTable and box" ;
    }
    if(!QObject::connect(save, SIGNAL(clicked()),this, SLOT(geInformation()),Qt::DirectConnection)){
        //qDebug << "[WARN] Can't connect EditorTable and box" ;
    }
    if(!QObject::connect(spinz, SIGNAL(valueChanged(int)),this, SLOT(changeZ(int)),Qt::DirectConnection)){
        //qDebug << "[WARN] Can't connect EditorTable and box" ;
    }


    QHBoxLayout* bLay = new QHBoxLayout();
    bLay->addWidget(buttonaddx);
    bLay->addWidget(buttonaddy);
    bLay->addWidget(buttonaddz);
    QHBoxLayout* bLay1 = new QHBoxLayout();
    bLay1->addWidget(buttonrmx);
    bLay1->addWidget(buttonrmy);
    bLay1->addWidget(buttonrmz);

    QHBoxLayout* bLay2 = new QHBoxLayout();
    bLay2->addWidget(spinz);
    bLay2->addWidget(save);

    QVBoxLayout *lay = new QVBoxLayout;
    lay->addLayout(bLay);
    lay->addLayout(bLay1);
    lay->addLayout(bLay2);


    lay->addWidget(box);

    test = false;
    this->setLayout(lay);
}
CControl * ControlEditorMatN3D::clone(){
    return new ControlEditorMatN3D;
}

string ControlEditorMatN3D::toString(){
    changeZ(_slice);

    std::ostringstream oss2;
    pop::MatNInOutPgm::writeAscii(oss2,img);
    cout<<oss2.str()<<endl;
    return oss2.str();
}
void ControlEditorMatN3D::fromString(string str){
    std::istringstream iss(str);
    cout<<str<<endl;
    MatNInOutPgm::read(iss,img);
    while(box->columnCount()!=img.getDomain()(0)){
        if(box->columnCount()<img.getDomain()(0)){
            addXTable();
        }else{
            rmXTable();
        }
    }
    while(box->rowCount()!=img.getDomain()(1)){
        if(box->rowCount()<img.getDomain()(1)){
            addYTable();
        }else{
            rmYTable();
        }
    }
     spinz->setMaximum(img.getDomain()(2)-1);

    VecN<3,int> x;
    int rcount = box->rowCount();
    int ccount = box->columnCount();
    for(x(1)=0;x(1)<rcount;x(1)++){
        for(x(0)=0;x(0)<ccount;x(0)++){
            QTableWidgetItem* item = box->item(x(1),x(0));
            x(2)=_slice;
            item->setText(UtilityString::Any2String(img(x)).c_str());
        }
    }
}

void ControlEditorMatN3D:: apply(){
    changeZ(_slice);
    if(test==true)
    {      
        string path = CMachineSingleton::getInstance()->getTmpPath()+"/"+UtilityString::Any2String(this->getKey())+UtilityString::Any2String(this->getId())+".pgm";
        DataMatN * data = new DataMatN;
        data->setFile(path);
        MatN<3,pop::F64> * temp = new MatN<3,pop::F64>(img);
        shared_ptr<BaseMatN> st(temp);
        data->setData(st);
        this->sendPlugOutControl(0,data,CPlug::NEW);
    }
}


void ControlEditorMatN3D::addX(){
    VecN<3,int> d = img.getDomain();
    d(0)++;
    img.resize(d);
    addXTable();
    update();
}

void ControlEditorMatN3D::addY(){
    VecN<3,int> d = img.getDomain();
    d(1)++;
    img.resize(d);
    addYTable();
    update();
}

void ControlEditorMatN3D::addZ(){
    VecN<3,int> d = img.getDomain();
    d(2)++;
    img.resize(d);
    spinz->setMaximum(d(2)-1);

}

void ControlEditorMatN3D::rmX(){
    VecN<3,int> d = img.getDomain();
    if(d(0)>1){
        d(0)--;
        img.resize(d);
        rmXTable();
        update();
    }
}

void ControlEditorMatN3D::rmY(){
    VecN<3,int> d = img.getDomain();
    if(d(1)>1){
        d(1)--;
        img.resize(d);
        rmYTable();
        update();
    }
}

void ControlEditorMatN3D::rmZ(){
    VecN<3,int> d = img.getDomain();
    if(d(2)>1){
        d(2)--;
        img.resize(d);
    }

}

void ControlEditorMatN3D::geInformation(){
    apply();
}

void ControlEditorMatN3D::changeZ(int z){
    VecN<3,int> x;
    x(2)=z;
    int rcount = box->rowCount();
    int ccount = box->columnCount();
    for(x(1)=0;x(1)<rcount;x(1)++){
        for(x(0)=0;x(0)<ccount;x(0)++){
            QTableWidgetItem* item = box->item(x(1),x(0));
            x(2)=_slice;
            img(x)= item->text().toDouble();
            x(2)=z;
            item->setText(UtilityString::Any2String(img(x)).c_str());
        }
    }
    _slice=z;
    cout<<img<<endl;
    update();
}
void ControlEditorMatN3D::addXTable(){
    int count = box->columnCount();
    box->setColumnCount(count+1);

    for(int i = 0;i<box->rowCount();i++){
        QTableWidgetItem* item = new QTableWidgetItem("0");
        box->setItem(i,count,item);
    }
    string cName = "X"+UtilityString::Any2String(count);
    QTableWidgetItem* itemName = new QTableWidgetItem(cName.c_str());
    box->setHorizontalHeaderItem(count,itemName);
}

void ControlEditorMatN3D::addYTable(){
    int count = box->rowCount();
    box->setRowCount(count+1);
    for(int i = 0;i<box->columnCount();i++){
        QTableWidgetItem* item = new QTableWidgetItem("0");
        box->setItem(count,i,item);
    }
    string cName = "Y"+UtilityString::Any2String(count);
    QTableWidgetItem* itemName = new QTableWidgetItem(cName.c_str());
    box->setVerticalHeaderItem(count,itemName);

}

void ControlEditorMatN3D::rmXTable(){
    int count = box->columnCount();
    if(count>0){
        box->setColumnCount(count-1);
        this->update();
    }

}

void ControlEditorMatN3D::rmYTable(){
    int count = box->rowCount();
    if(count>0){
        box->setRowCount(count-1);
        this->update();
    }
}

