#ifndef XMLDOCUMENT_H
#define XMLDOCUMENT_H
#include <iostream>
#include <fstream>
#include <sstream>
#include "PopulationConfig.h"
#include "3rdparty/pugixml.hpp"
namespace pop
{

class POP_EXPORTS XMLNode
{
public:
    XMLNode();
    ~XMLNode();
    XMLNode(const XMLNode & node);
    XMLNode& operator ()(const XMLNode & node);
    void setName(std::string name);
    void setValue(std::string value);
    XMLNode addChild(std::string name);
    bool rmChild(const XMLNode& node);
    std::string getName()const;
    std::string getValue()const;
    XMLNode getChild(std::string childname)const;
    XMLNode firstChild()const;
    /*!
    \code
    XMLDocument doc;
    doc.load("/home/tariel/Bureau/camera.xml");
    XMLNode node = doc.getChild("CAMERA");
    for (XMLNode tool = node.firstChild(); tool; tool = tool.nextSibling())
    {
        std::cout << "IP:";
        std::cout<<tool.getAttribute("IP")<<std::endl;
        std::cout << std::endl;
    }
    \endcode
    The file camera.xml is:
    \code
<?xml version="1.0" encoding="UTF-8"?>
<CAMERA>
  <camera IP="rtsp://193.222.150.5/channel1" frame_rate="100" channel="0" name="camera entree" type="0" />
  <camera IP="rtsp://193.222.150.105/channel1" frame_rate="200" channel="1" name="camera sortie" type="0" />
</CAMERA>
     \endcode
    */
    operator bool() const ;
//    xml_node::operator xml_node::unspecified_bool_type() const


    XMLNode nextSibling()const;
    std::string getAttribute(std::string attributename) const;
    bool hasAttribute(std::string attributename)const;
    void setAttribute(std::string name,std::string  value);
    void addAttribute(std::string name,std::string  value);
    bool rmAttribute(std::string name);
//    struct Impl;
//    Impl *_impl;
    pugi::xml_node _node;
};


class POP_EXPORTS XMLDocument
{

public:

  XMLDocument();
  XMLDocument(const XMLDocument& doc);
  XMLDocument& operator =(const XMLDocument& doc);
  ~XMLDocument();
  void load(std::string file);
  void loadFromByteArray(const char *file);
  void save(std::string file)const ;
  XMLNode getChild(std::string name);
  XMLNode addChild(std::string name);
  struct Impl;
private:
  Impl *_impl;
  };

}
#endif // XMLDOCUMENT_H
