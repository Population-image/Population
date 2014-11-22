#ifndef CONTROLEDITORPOINT_H
#define CONTROLEDITORPOINT_H

#include<EditorTable.h>

class ControlEditorPoint : public EditorTable
{
public:
    ControlEditorPoint(QWidget * parent = 0);
    void apply();
    virtual CControl * clone();
//        virtual QSize 	minimumSizeHint() const;
protected:
    virtual QString defaultString();
    virtual QString defaultHeaderColumn(int col);
    virtual QString defaultHeaderRaw(int raw);
};

#endif // CONTROLEDITORPOINT_H
