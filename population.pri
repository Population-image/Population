INCLUDEPATH += $${PWD}/include/

HEADERS += $${PWD}/include/Population.h \
           $${PWD}/include/PopulationConfig.h \
           $${PWD}/include/algorithm/Analysis.h \
           $${PWD}/include/algorithm/AnalysisAdvanced.h \
           $${PWD}/include/algorithm/Application.h \
           $${PWD}/include/algorithm/Convertor.h \
           $${PWD}/include/algorithm/Draw.h \
           $${PWD}/include/algorithm/Feature.h \
           $${PWD}/include/algorithm/ForEachFunctor.h \
           $${PWD}/include/algorithm/GeometricalTransformation.h \
           $${PWD}/include/algorithm/LinearAlgebra.h \
           $${PWD}/include/algorithm/PDE.h \
           $${PWD}/include/algorithm/PDEAdvanced.h \
           $${PWD}/include/algorithm/Processing.h \
           $${PWD}/include/algorithm/ProcessingAdvanced.h \
           $${PWD}/include/algorithm/ProcessingVideo.h \
           $${PWD}/include/algorithm/RandomGeometry.h \
           $${PWD}/include/algorithm/Representation.h \
           $${PWD}/include/algorithm/Statistics.h \
           $${PWD}/include/algorithm/Visualization.h \
           $${PWD}/include/dependency/bipmap.h \
           $${PWD}/include/dependency/CImg.h \
           $${PWD}/include/dependency/ConvertorCImg.h \
           $${PWD}/include/dependency/ConvertorOpenCV.h \
           $${PWD}/include/dependency/ConvertorQImage.h \
           $${PWD}/include/dependency/direntvc.h \
           $${PWD}/include/dependency/fparser.hh \
           $${PWD}/include/dependency/fpconfig.hh \
           $${PWD}/include/dependency/fptypes.hh \
           $${PWD}/include/dependency/jpgd.h \
           $${PWD}/include/dependency/jpge.h \
           $${PWD}/include/dependency/lodepng.h \
           $${PWD}/include/dependency/log.h \
           $${PWD}/include/dependency/MatNDisplayCImg.h \
           $${PWD}/include/dependency/MTRand.h \
           $${PWD}/include/dependency/pugiconfig.hpp \
           $${PWD}/include/dependency/pugixml.hpp \
           $${PWD}/include/dependency/tinythread.h \
           $${PWD}/include/dependency/VideoFFMPEG.h \
           $${PWD}/include/dependency/VideoVLC.h \
           $${PWD}/include/data/3d/GLFigure.h \
           $${PWD}/include/data/distribution/Distribution.h \
           $${PWD}/include/data/distribution/DistributionAnalytic.h \
           $${PWD}/include/data/distribution/DistributionArithmetic.h \
           $${PWD}/include/data/distribution/DistributionFromDataStructure.h \
           $${PWD}/include/data/distribution/DistributionMultiVariate.h \
           $${PWD}/include/data/distribution/DistributionMultiVariateArithmetic.h \
           $${PWD}/include/data/distribution/DistributionMultiVariateFromDataStructure.h \
           $${PWD}/include/data/functor/FunctorF.h \
           $${PWD}/include/data/functor/FunctorPDE.h \
           $${PWD}/include/data/germgrain/Germ.h \
           $${PWD}/include/data/germgrain/GermGrain.h \
           $${PWD}/include/data/GP/CartesianProduct.h \
           $${PWD}/include/data/GP/Dynamic2Static.h \
           $${PWD}/include/data/GP/EmptyType.h \
           $${PWD}/include/data/GP/Factory.h \
           $${PWD}/include/data/GP/LokiTypeInfo.h \
           $${PWD}/include/data/GP/NullType.h \
           $${PWD}/include/data/GP/Singleton.h \
           $${PWD}/include/data/GP/TTypeTraits.h \
           $${PWD}/include/data/GP/Type2Id.h \
           $${PWD}/include/data/GP/Typelist.h \
           $${PWD}/include/data/GP/TypelistMacros.h \
           $${PWD}/include/data/GP/TypeManip.h \
           $${PWD}/include/data/GP/TypeTraitsTemplateTemplate.h \
           $${PWD}/include/data/mat/Mat2x.h \
           $${PWD}/include/data/mat/MatN.h \
           $${PWD}/include/data/mat/MatNBoundaryCondition.h \
           $${PWD}/include/data/mat/MatNDisplay.h \
           $${PWD}/include/data/mat/MatNInOut.h \
           $${PWD}/include/data/mat/MatNIteratorE.h \
           $${PWD}/include/data/mat/MatNListType.h \
           $${PWD}/include/data/neuralnetwork/NeuralNetwork.h \
           $${PWD}/include/data/notstable/CharacteristicCluster.h \
           $${PWD}/include/data/notstable/Classifer.h \
           $${PWD}/include/data/notstable/Descriptor.h \
           $${PWD}/include/data/notstable/Ransac.h \
           $${PWD}/include/data/notstable/SearchStructure.h \
           $${PWD}/include/data/notstable/Wavelet.h \
           $${PWD}/include/data/ocr/OCR.h \
           $${PWD}/include/data/population/PopulationData.h \
           $${PWD}/include/data/population/PopulationFunctor.h \
           $${PWD}/include/data/population/PopulationGrows.h \
           $${PWD}/include/data/population/PopulationPDE.h \
           $${PWD}/include/data/population/PopulationQueues.h \
           $${PWD}/include/data/population/PopulationRestrictedSet.h \
           $${PWD}/include/data/typeF/Complex.h \
           $${PWD}/include/data/typeF/RGB.h \
           $${PWD}/include/data/typeF/TypeF.h \
           $${PWD}/include/data/typeF/TypeTraitsF.h \
           $${PWD}/include/data/utility/BasicUtility.h \
           $${PWD}/include/data/utility/Cryptography.h \
           $${PWD}/include/data/utility/BSPTree.h \
           $${PWD}/include/data/utility/CollectorExecutionInformation.h \
           $${PWD}/include/data/utility/Exception.h \
           $${PWD}/include/data/utility/XML.h \
           $${PWD}/include/data/vec/Vec.h \
           $${PWD}/include/data/vec/VecN.h \
           $${PWD}/include/data/video/Video.h \
           $${PWD}/include/data/notstable/graph/Graph.h \
           $$PWD/include/data/functor/FunctorMatN.h

SOURCES += $${PWD}/src/algorithm/GeometricalTransformation.cpp \
           $${PWD}/src/algorithm/LinearAlgebra.cpp \
           $${PWD}/src/algorithm/RandomGeometry.cpp \
           $${PWD}/src/algorithm/Statistics.cpp \
           $${PWD}/src/algorithm/Visualization.cpp \
           $${PWD}/src/dependency/ConvertorOpenCV.cpp \
           $${PWD}/src/dependency/ConvertorQImage.cpp \
           $${PWD}/src/dependency/fparser.cpp \
           $${PWD}/src/dependency/jpgd.cpp \
           $${PWD}/src/dependency/jpge.cpp \
           $${PWD}/src/dependency/lodepng.cpp \
           $${PWD}/src/dependency/MatNDisplayCImg.cpp \
           $${PWD}/src/dependency/MTRand.cpp \
           $${PWD}/src/dependency/pugixml.cpp \
           $${PWD}/src/dependency/tinythread.cpp \
           $${PWD}/src/dependency/VideoFFMPEG.cpp \
           $${PWD}/src/dependency/VideoVLC.cpp \
           $${PWD}/src/data/3d/GLFigure.cpp \
           $${PWD}/src/data/distribution/Distribution.cpp \
           $${PWD}/src/data/distribution/DistributionAnalytic.cpp \
           $${PWD}/src/data/distribution/DistributionArithmetic.cpp \
           $${PWD}/src/data/distribution/DistributionFromDataStructure.cpp \
           $${PWD}/src/data/distribution/DistributionMultiVariate.cpp \
           $${PWD}/src/data/distribution/DistributionMultiVariateArithmetic.cpp \
           $${PWD}/src/data/distribution/DistributionMultiVariateFromDataStructure.cpp \
           $${PWD}/src/data/germgrain/GermGrain.cpp \
           $${PWD}/src/data/mat/MatNDisplay.cpp \
           $${PWD}/src/data/mat/MatNInOut.cpp \
           $${PWD}/src/data/mat/MatNListType.cpp \
           $${PWD}/src/data/neuralnetwork/NeuralNetwork.cpp \
           $${PWD}/src/data/notstable/CharacteristicCluster.cpp \
           $${PWD}/src/data/notstable/Ransac.cpp \
           $${PWD}/src/data/ocr/OCR.cpp \
           $${PWD}/src/data/typeF/TypeTraitsF.cpp \
           $${PWD}/src/data/utility/BasicUtility.cpp \
           $${PWD}/src/data/utility/Cryptography.cpp \
           $${PWD}/src/data/utility/CollectorExecutionInformation.cpp \
           $${PWD}/src/data/utility/Exception.cpp \
           $${PWD}/src/data/utility/XML.cpp \
           $${PWD}/src/data/video/Video.cpp

#Path to the project
DEFINES += 'POP_PROJECT_SOURCE_DIR=\'\"$${PWD}\"\''
#Do not include the popconfig.h generating by cmake
DEFINES+=HAVE_QMAKE

HAVE_OPENGL{
    DEFINES+=HAVE_OPENGL
    DEFINES+=HAVE_THREAD
    unix:LIBS+=-lglut -lGL -lGLU  -lpthread
    win32:LIBS += -lAdvapi32 -lgdi32 -luser32 -lshell32 -lopengl32 -lglu32
}
HAVE_CIMG{
    DEFINES+=HAVE_CIMG
    DEFINES*=HAVE_THREAD
    unix:LIBS*=-lX11 -lpthread
    win32:LIBS*=-lAdvapi32 -lgdi32 -luser32 -lshell32
}

HAVE_VLC{
    DEFINES+=HAVE_VLC
    DEFINES*=HAVE_THREAD
    unix:LIBS*=-lX11 -lpthread
    win32:LIBS*=-lAdvapi32 -lgdi32 -luser32 -lshell32
    win32:INCLUDEPATH +="C:/Program Files (x86)/VideoLAN/VLC/sdk/include/"
    win32:LIBS += -L"C:/Program Files (x86)/VideoLAN/VLC"
    win32-msvc2010 {
        LIBS += -llibvlc
    } else {
        LIBS += -lvlc
    }
}
HAVE_QT {
    CONFIG*=qt
    DEFINES+= HAVE_QT
}
HAVE_FFMPEG {
    DEFINES+= HAVE_FFMPEG
    DEFINES*=HAVE_THREAD
    unix:LIBS*=-lX11 -lpthread
    win32:LIBS*=-lAdvapi32 -lgdi32 -luser32 -lshell32
    win32:INCLUDEPATH+=$${PWD}/core/dependency/ffmpeg/include
    win32:LIBS+=-L$${PWD}/core/dependency/ffmpeg/lib
    LIBS += -lavcodec
    LIBS += -lavformat
    LIBS += -lavutil
    LIBS += -lswscale
}

HAVE_OPENCV {
    DEFINES+= HAVE_OPENCV
    unix:CONFIG += link_pkgconfig
    unix:PKGCONFIG += opencv
}
HAVE_OPENMP {
    DEFINES+= HAVE_OPENMP
    unix:QMAKE_CXXFLAGS+= -fopenmp
    unix:QMAKE_LFLAGS +=  -fopenmp
    win32:QMAKE_CXXFLAGS+= -openmp
    win32:QMAKE_LFLAGS +=  -openmp
}else {
    QMAKE_CXXFLAGS+= -Wunknown-pragmas
 }
