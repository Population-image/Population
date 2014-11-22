TEMPLATE = subdirs

SUBDIRS += ../../populationapp.pro

#generate shared library
SUBDIRS += population.pro
SUBDIRS += populationsharedlibrarytest.pro

#Articles
SUBDIRS += ../article/germgrain/germgrain.pro

#generate python
SUBDIRS +=  ../python/populationpython.pro



CONFIG += ordered
