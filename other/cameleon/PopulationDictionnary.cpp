
#include "PopulationDictionnary.h"
#include "DataImageGrid.h"
#include "DataDistribution.h"
#include "DataPoint.h"
#include "DataMatrix.h"
#include "DataGraph.h"
#include "DataGrainList.h"
#include "DataOpenGl.h"
//Control
#include "ControlEditorPoint.h"
#include "ControlEditorMatrix.h"
#include "ControlViewOpenGl.h"
#include "ControlViewImageGrid.h"
#include "ControlEditorImageGrid.h"
#include "ControlMarkerImageGrid.h"
#include "ControlViewImageGridValue.h"
#include "ControlViewMatrix.h"
#include "ControlViewPoint.h"



//Generator
#include "OperatorConstImageGrid.h"
#include "OperatorRandomFieldImageGrid.h"


//Input/output

#include "OperatorIsEmptyImageGrid.h"
#include "OperatorLoadImageGrid.h"
#include "OperatorLoadRawImageGrid.h"
#include "OperatorSaveImageGrid.h"
#include "OperatorLoadFromDirectoryImageGrid.h"
#include "OperatorSaveFromDirectoryImageGrid.h"
#include "OperatorIsReadableImageGrid.h"
#include "OperatorLoadRawImageGrid.h"
#include "OperatorSaveRawImageGrid.h"

//Format
#include "OperatorConvert1ByteImageGrid.h"
#include "OperatorConvert2ByteImageGrid.h"
#include "OperatorConvert4ByteImageGrid.h"
#include "OperatorConvertFloatImageGrid.h"
#include "OperatorConvertColor2GreyImageGrid.h"
#include "OperatorConvertGrey2ColorImageGrid.h"
#include "OperatorConvertColor2RGBImageGrid.h"
#include "OperatorConvertRGB2ColorImageGrid.h"
#include "OperatorConvertScalar2ComplexImageGrid.h"
#include "OperatorConvertComplex2ScalarImageGrid.h"

#include "OperatorConvertImage3DToVector.h"
#include "OperatorConvertVectorToImage3D.h"


#include "OperatorConvertImageGrid2ImageQT.h"
#include "OperatorConvertImageQT2ImageGrid.h"

//Tool
#include "OperatorGetSizeImageGrid.h"
#include "OperatorResizeImageGrid.h"
#include "OperatorTypeImageGrid.h"
#include "OperatorLabelOrganizeImageGrid.h"
#include "OperatorLabelToImageGridVectorImageGrid.h"
#include "OperatorGetPlaneImageGrid.h"
#include "OperatorSetPlaneImageGrid.h"
#include "OperatorGetPointImageGrid.h"
#include "OperatorSetPointImageGrid.h"
#include "OperatorSetBallImageGrid.h"

#include "OperatorHistogramShiftMeanImageGrid.h"
#include "OperatorScaleDynamicGreyLevelImageGrid.h"
//Point
//Unary
#include "OperatorThresholdImageGrid.h"
#include "OperatorfofxImageGrid.h"
#include "OperatorContrastScaleImageGrid.h"
#include "OperatorHeteregeneousRandomFieldImageGrid.h"

//Binary
#include "OperatorAddImageGrid.h"
#include "OperatorSubImageGrid.h"
#include "OperatorMaxImageGrid.h"
#include "OperatorMinImageGrid.h"
#include "OperatorMultImageGrid.h"
#include "OperatorDivImageGrid.h"
#include "OperatorMinuxImageGrid.h"
#include "OperatorDiffImageGrid.h"
#include "OperatorAddScalarImageGrid.h"
#include "OperatorSubScalarImageGrid.h"
#include "OperatorMultScalarImageGrid.h"
#include "OperatorDivScalarImageGrid.h"
//neighborhood
#include "OperatorErosionImageGrid.h"
#include "OperatorErosionFastBinaryImageGrid.h"
#include "OperatorDilationImageGrid.h"
#include "OperatorDilationFastBinaryImageGrid.h"
#include "OperatorOpeningImageGrid.h"
#include "OperatorClosingImageGrid.h"
#include "OperatorAlternateSequentialCOImageGrid.h"
#include "OperatorAlternateSequentialOCImageGrid.h"
#include "OperatorMedianImageGrid.h"
#include "OperatorMeanImageGrid.h"

#include "OperatorErosionStructuralImageGrid.h"
#include "OperatorDilationStructuralImageGrid.h"
#include "OperatorOpeningStructuralImageGrid.h"
#include "OperatorClosingStructuralImageGrid.h"
#include "AlternateSequentialCOStructural.h"
#include "AlternateSequentialOCStructural.h"
#include "OperatorHitOrMissImageGrid.h"
//convolution
#include "OperatorConvolutionImageGrid.h"
#include "OperatorSobelImageGrid.h"
#include "OperatorSmoothGaussianImageGrid.h"
#include "OperatorGradientMagnitudeGaussianImageGrid.h"
//recursive
#include "OperatorRecursiveOrder1ImageGrid.h"
#include "OperatorRecursiveOrder2ImageGrid.h"
#include "OperatorGradientDericheImageGrid.h"
#include "OperatorSmoothDericheImageGrid.h"
//region growing
#include "OperatorLabelAddImageGrid.h"
#include "OperatorBinaryFromLabelSelectionImageGrid.h"
#include "OperatorBinaryFromSingleSeedInLabelImageGrid.h"


#include "OperatorCluster2LabelImageGrid.h"
#include "OperatorClusterMaxImageGrid.h"
#include "OperatorVoronoiTesselationImageGrid.h"
#include "OperatorGeodesicReconstructionImageGrid.h"
#include "OperatorDynamicImageGrid.h"
#include "OperatorWatershedImageGrid.h"
#include "OperatorDistanceImageGrid.h"
#include "OperatorMinimaImageGrid.h"
#include "OperatorAdamsBischofImageGrid.h"
#include "OperatorHoleFillingImageGrid.h"
//PDE
#include "OperatorNonLinearAnistropicDiffusionImageGrid.h"
#include "OperatorAllenCahnImageGrid.h"
#include "OperatorCurvatureFromPhaseFieldImageGrid.h"
#include "OperatorGetFieldFromMultiPhaseFieldImageGrid.h"

//visualization
//2D
#include "OperatorColorAverageImageGrid.h"
#include "OperatorColorForegroundImageGrid.h"
#include "OperatorColorGradationFromLabelImageGrid.h"
#include "OperatorColorRandomFromLabelImageGrid.h"


//3D
#include "OperatorAdditionOpengl.h"
#include "OperatorCubeOpengl.h"
#include "OperatorSetColorOpengl.h"
#include "OperatorGetPlaneOpengl.h"
#include "OperatorGraphOpenGl.h"
#include "OperatorLatticeSurfaceOpenGl.h"
#include "OperatorLineOpengl.h"
#include "OperatorMarchingCubeBinaryImageGridOpenGl.h"
#include "OperatorMarchingCubeColorImageGridOpenGl.h"
#include "OperatorTransparencyOpengl.h"
#include "OperatorSetAmbientOpengl.h"
#include "operatorsetdiffuseopengl.h"
//Geometry
#include "OperatorCropImageGrid.h"
#include "OperatorInsertImageImageGrid.h"
#include "OperatorScaleImageGrid.h"
#include "OperatorTranslationImageGrid.h"
#include "OperatorRotationImageGrid.h"
#include "OperatorBlankImageGrid.h"
#include "OperatorWhiteFaceImageGrid.h"

//GermGrain
//Germ
#include "OperatorHardCoreFilterGrainList.h"
#include "OperatorMinOverlapGrainList.h"
#include "OperatorRandomNonUniformPointGrainList.h"
#include "OperatorRandomUniformPointGrainList.h"
//Grain
#include "OperatorBoxGrainList.h"
#include "OperatorCylinderGrainList.h"
#include "OperatorEllipsoidGrainList.h"

#include "OperatorPolyhedraGrainList.h"

#include "OperatorRhombohedronGrainList.h"
#include "OperatorSphereGrainList.h"
#include "OperatorGrainFromImageGrainList.h"
//Lattice
#include "OperatorContinuousToLatticeGrainList.h"
//Model
#include "OperatorBooleanGrainList.h"
#include "OperatorDeadLeaveGrainList.h"
#include "OperatorTransparencyGrainList.h"
//AffectColor
#include "OperatorColorFromImageGrainList.h"
#include "OperatorRandomBlackOrWhiteGrainList.h"
#include "OperatorRandomColorGrainList.h"
//Art
#include "OperatorRandomWalkGrainList.h"

//Geometrical constraint
#include "OperatorGaussianRandomFieldImageGrid.h"
#include "OperatorRandomStructureImageGrid.h"
#include "OperatorAnnealingSimulatedImageGrid.h"

//Analysis
//Morphology

#include "OperatorVERPointPorosityImageGrid.h"
#include "OperatorVERPointHistogramImageGrid.h"
//Scalar
#include "OperatorAreaImageGrid.h"
#include "OperatorHistogramImageGrid.h"
#include "OperatorMaxValueImageGrid.h"
#include "OperatorMinValueImageGrid.h"
#include "OperatorPerimeterImageGrid.h"
//Statistic
#include "OperatorChordImageGrid.h"
#include "OperatorCorrelationImageGrid.h"
#include "OperatorCorrelationGreyLevelImageGrid.h"
#include "OperatorCorrelationDirectionByFFTImageGrid.h"
#include "OperatorFractalBoxImageGrid.h"
#include "OperatorMatheronGranulometryImageGrid.h"
#include "OperatorLDistanceImageGrid.h"
//Label
#include "OperatorRepartitionAreaLabelImageGrid.h"
#include "OperatorRepartitionPerimeterLabelImageGrid.h"
#include "OperatorRepartitionFeretDiameterLabelImageGrid.h"
#include "OperatorRepartitionContactLabelImageGrid.h"
//Topology
//Scalar
#include "OperatorPercolationImageGrid.h"
#include "OperatorPercolationErosionImageGrid.h"
#include "OperatorPercolationOpeningImageGrid.h"
#include "OperatorEulerPoincareImageGrid.h"
#include "OperatorGeometricalTortuosityImageGrid.h"
//skeleton
#include "OperatorMedialAxisImageGrid.h"
#include "OperatorThinningAtConstantTopology.h"
#include "OperatorVertexAndEdgeFromSkeletonImageGrid.h"
#include "OperatorLinkVertexWithEdgeImageGrid.h"
//Physical
#include "OperatorPermeabilityImageGrid.h"
#include "OperatorDiffusionCoefficientImageGrid.h"
//Representation
#include "OperatorFFTImageGrid.h"
#include "OperatorFFTInverseImageGrid.h"
#include "OperatorVertexAndEdgeFromSkeletonImageGrid.h"

//Distribution
//Convert
#include "OperatorSaveDistribution.h"
#include "OperatorLoadDistribution.h"
#include "OperatorConvertProbabilityDistributionDistribution.h"
#include "OperatorConvertStepFunctionDistribution.h"
#include "OperatorToMatrixDistribution.h"
#include "OperatorConvertProbabilityDistributionDistribution.h"
#include "OperatorConvertCumulativeDistribution.h"
#include "OperatorConvertStepFunctionDistribution.h"
#include "OperatorToMatrixDistribution.h"



//Continuous
#include "OperatorUniformRealDistribution.h"
#include "OperatorBoxDistribution.h"
#include "OperatorExponentielDistribution.h"
#include "OperatorNormalDistribution.h"
#include "OperatorDiracDistribution.h"
#include "OperatorRegularExpressionDistribution.h"
#include "OperatorFromMatrixDistribution.h"
//Discrete
#include "OperatorUniformIntDistribution.h"
#include "OperatorPoissonDistribution.h"
#include "OperatorBinomialDistribution.h"
#include "OperatorPencilDistribution.h"
//Arithmetic
#include "OperatorAddDistribution.h"
#include "OperatorCompoDistribution.h"
#include "OperatorDivDistribution.h"
#include "OperatorInverseDistribution.h"
#include "OperatorMultDistribution.h"
#include "OperatorOppositeDistribution.h"
#include "OperatorSubDistribution.h"
#include "OperatorMaxDistribution.h"
//OPERATOR
#include "OperatorIntegralDistribution.h"
#include "OperatorDerivateDistribution.h"
#include "OperatorfofXDistribution.h"
#include "OperatorArgMaxDistribution.h"
#include "OperatorArgMinDistribution.h"
#include "OperatorMaxValueDistribution.h"
#include "OperatorMinValueDistribution.h"
#include "OperatorRandomVariableDistribution.h"
#include "OperatorMomentDistribution.h"
#include "OperatorDistanceDistribution.h"
#include "OperatorComputedStaticticsDistribution.h"


#include "OperatorIntegralDistributionMultiVariate.h"
#include "OperatorArgMaxDistributionMultiVariate.h"
#include "OperatorArgMinDistributionMultiVariate.h"
#include "OperatorMaxValueDistributionMultiVariate.h"
#include "OperatorMinValueDistributionMultiVariate.h"
#include "OperatorMomentDistributionMultiVariate.h"
#include "OperatorConvertProbabilityDistributionMultiVariateDistributionMultiVariate.h"
#include "OperatorFromDistributionDistributionMultiVariate.h"
//DistributionmultiVariate

#include "DataDistributionMultiVariate.h"
//InOut
#include "OperatorSaveDistributionMultiVariate.h"
#include "OperatorLoadDistributionMultiVariate.h"
#include "OperatorTryLoad.h"

#include "OperatorNormalDistributionMultiVariate.h"
#include "OperatorExpressionDistributionMultiVariate.h"
#include "OperatorCoupledDistributionMultiVariate.h"
#include "OperatorIndependantDistributionMultiVariate.h"


//Arithmetic
#include "OperatorAddDistributionMultiVariate.h"
#include "OperatorCompoDistributionMultiVariate.h"
#include "OperatorDivDistributionMultiVariate.h"
#include "OperatorInverseDistributionMultiVariate.h"
#include "OperatorMultDistributionMultiVariate.h"
#include "OperatorOppositeDistributionMultiVariate.h"
#include "OperatorSubDistributionMultiVariate.h"
#include "OperatorMaxDistributionMultiVariate.h"

#include "OperatorRandomVariableDistributionMultiVariate.h"
#include "OperatorfofXDistributionMultiVariate.h"


//Linear Algebra
#include "OperatorAddMatrix.h"
#include "OperatorAddVector.h"
#include "OperatorAddScalarVector.h"
#include "OperatorSubMatrix.h"
#include "OperatorSubVector.h"

#include "OperatorMultMatrix.h"
#include "OperatorMultVector.h"
#include "OperatorMultEachTermVector.h"
#include "OperatorMultMatrixScalar.h"
#include "OperatorMultVectorScalar.h"

#include "OperatorMultVectorMatrix.h"
#include "OperatorMinVector.h"
#include "OperatorMaxVector.h"

#include "OperatorLoadMatrix.h"
#include "OperatorSaveMatrix.h"
#include "OperatorLoadPoint.h"
#include "OperatorSavePoint.h"
#include "OperatorBlankMatrix.h"
#include "OperatorBlankVector.h"
#include "OperatorBlank2DVector.h"
#include "operatorblank3dvector.h"

#include "OperatorConvertToTableMatrix.h"
#include "OperatorConvertFromTableMatrix.h"

#include "OperatorToImageGridMatrix.h"
#include "OperatorFromImageGridMatrix.h"
#include "OperatorSetMatrix.h"
#include "OperatorGetMatrix.h"
#include "OperatorResizeMatrix.h"
#include "OperatorSizeMatrix.h"
#include "OperatorSetPoint.h"
#include "OperatorGetPoint.h"
#include "OperatorPushBackValueVector.h"
#include "OperatorMultCoordinateVector.h"
#include "OperatorResizePoint.h"
#include "OperatorSizePoint.h"
#include "OperatorSetColMatrix.h"
#include "OperatorSetRawMatrix.h"
#include "OperatorGetColMatrix.h"
#include "OperatorGetRawMatrix.h"
#include "OperatorGenerate2DRotationMatrix.h"
#include "OperatorGenerate3DRotationMatrix.h"

#include "OperatorDeterminantMatrix.h"
#include "OperatorTraceMatrix.h"
#include "OperatorTransposeMatrix.h"
#include "OperatorInverseMatrix.h"
#include "OperatorIdentityMatrix.h"
#include "OperatorOrthogonalMatrix.h"
#include "OperatorEigenValueMatrix.h"
#include "OperatorEigenVectorMatrix.h"



//Plot
#include "DataPlot.h"
#include "OperatorAddGraphPlot.h"
#include "OperatorBlankPlot.h"
#include "OperatorFromMatrixPlot.h"
#include "OperatorFromDistributionPlot.h"
#include "OperatorFromTablePlot.h"
#include "OperatorPopPointPlot.h"
#include "OperatorPushPointPlot.h"
#include "OperatorSetColorPlot.h"
#include "OperatorSetLegendPlot.h"
#include "OperatorSetTitlePlot.h"
#include "OperatorSetWidthPlot.h"
#include "OperatorSetAlphaPlot.h"
#include "OperatorSetBrushColorPlot.h"
#include "OperatorSetXAxisLegendPlot.h"
#include "OperatorSetYAxisLegendPlot.h"
#include "OperatorSetLogXAxisPlot.h"
#include "OperatorSetLogYAxisPlot.h"


#include "PopulationLog.h"
#include"data/utility/CollectorExecutionInformation.h"


PopulationDictionnary::PopulationDictionnary(){
    this->setNameDictionnary("Population");
    this->setVersion("3.1.0");


    PopulationLog * collector =new PopulationLog;
    CollectorExecutionInformationSingleton::getInstance()->setCollector(collector);
    CollectorExecutionInformationSingleton::getInstance()->setActivate(true);
}

void PopulationDictionnary::collectData(){
    this->registerData(new DataMatN);
    this->registerData(new DataGraph);

    this->registerData(new DataDistribution);
    this->registerData(new DataDistributionMultiVariate);
    this->registerData(new DataPoint);
    this->registerData(new DataMatrix);


    this->registerData(new DataGermGrain);
    this->registerData(new DataOpenGl);

    this->registerData(new DataPlot);
}

void PopulationDictionnary::collectOperator(){


    //Input/Output
    this->registerOperator(new OperatorLoadMatN);
    this->registerOperator(new OperatorSaveMatN);
    this->registerOperator(new OperatorTryLoad);
    this->registerOperator(new OperatorLoadFromDirectoryMatN);
    this->registerOperator(new OperatorSaveFromDirectoryMatN);
    this->registerOperator(new OperatorIsReadableMatN);
    this->registerOperator(new OperatorLoadRawMatN);
    this->registerOperator(new OperatorSaveRawMatN);


    //Tool
    //Format
    this->registerOperator(new OperatorConvert1ByteMatN);
    this->registerOperator(new OperatorConvert2ByteMatN);
    this->registerOperator(new OperatorConvert4ByteMatN);
    this->registerOperator(new OperatorConvertFloatMatN);
    this->registerOperator(new OperatorConvertColor2GreyMatN);
    this->registerOperator(new OperatorConvertGrey2ColorMatN);
    this->registerOperator(new OperatorConvertColor2RGBMatN);
    this->registerOperator(new OperatorConvertRGB2ColorMatN);
    this->registerOperator(new OperatorConvertScalar2ComplexMatN);
    this->registerOperator(new OperatorConvertComplex2ScalarMatN);

    this->registerOperator(new OperatorMatN2ImageQT);
    this->registerOperator(new OperatorImageQT2MatN);
    this->registerOperator(new OperatorConvertImage3DToVector);
    this->registerOperator(new OperatorConvertVectorToImage3D);




    this->registerOperator(new OperatorGetSizeMatN);
    this->registerOperator(new OperatorResizeMatN);
    this->registerOperator(new OperatorTypeMatN);
    this->registerOperator(new OperatorLabelOrganizeMatN);


    this->registerOperator(new OperatorLabelToMatNVectorMatN);
    this->registerOperator(new OperatorGetPlaneMatN);
    this->registerOperator(new OperatorsetPlaneMatN);
    this->registerOperator(new OperatorGetPointMatN);
    this->registerOperator(new OperatorSetPointMatN);
    this->registerOperator(new OperatorSetBallMatN);
    this->registerOperator(new OperatorIsEmptyMatN);
    this->registerOperator(new OperatorHistogramShiftMeanmageGrid);
    this->registerOperator(new OperatorScaleDynamicGreyLevelMatN);
    //Generator
    this->registerOperator(new OperatorConstMatN);
    this->registerOperator(new OperatorRandomFieldMatN);

    //Point
    //Unary
    this->registerOperator(new OperatorThresholdMatN);
    this->registerOperator(new OperatorContrastScaleMatN);
    this->registerOperator(new OperatorContrastScaleColorMatN);
    this->registerOperator(new OperatorfofxMatN);
//    this->registerOperator(new OperatorHeteregeneousRandomFieldMatN);
    //Binary
    this->registerOperator(new OperatorAddMatN);
    this->registerOperator(new OperatorSubMatN);
    this->registerOperator(new OperatorMaxMatN);
    this->registerOperator(new OperatorMinMatN);
    this->registerOperator(new OperatorMultMatN);
    this->registerOperator(new OperatorDivMatN);
    this->registerOperator(new OperatorMinusMatN);
    this->registerOperator(new OperatorDiffMatN);

    this->registerOperator(new OperatorAddScalarMatN);
    this->registerOperator(new OperatorSubScalarMatN);
    this->registerOperator(new OperatorMultScalarMatN);
    this->registerOperator(new OperatorDivScalarMatN);


    //neighborhood
    //morphology
    this->registerOperator(new OperatorErosionMatN);
    this->registerOperator(new OperatorErosionFastBinaryMatN);
    this->registerOperator(new OperatorDilationMatN);
    this->registerOperator(new OperatorDilationFastBinaryMatN);
    this->registerOperator(new OperatorOpeningMatN);
    this->registerOperator(new OperatorClosingMatN);
    this->registerOperator(new OperatorAlternateSequentialCOMatN);
    this->registerOperator(new OperatorAlternateSequentialOCMatN);
    this->registerOperator(new OperatorMedianMatN);
    this->registerOperator(new OperatorMeanMatN);
    this->registerOperator(new OperatorErosionStructuralMatN);
    this->registerOperator(new OperatorDilationStructuralMatN);
    this->registerOperator(new OperatorOpeningStructuralMatN);
    this->registerOperator(new OperatorClosingStructuralMatN);
    this->registerOperator(new OperatorAlternateSequentialCOStructuralMatN);
    this->registerOperator(new OperatorAlternateSequentialOCStructuralMatN);
    this->registerOperator(new OperatorHitOrMissMatN);

    //kernel
    this->registerOperator(new OperatorConvolutionMatN);
    this->registerOperator(new OperatorGradientNormSobelMatN);
    this->registerOperator(new OperatorSmoothGaussianMatN);
    this->registerOperator(new OperatorGradientGaussianMatN);

    //recursive
    this->registerOperator(new OperatorRecursiveOrder1MatN);
    this->registerOperator(new OperatorRecursiveOrder2MatN);
    this->registerOperator(new OperatorGradientDericheMatN);
    this->registerOperator(new OperatorSmoothDericheMatN);
    //region growing
    //seed
    this->registerOperator(new OperatorLabelAddMatN);
    this->registerOperator(new OperatorBinaryFromLabelSelectionMatN);
    this->registerOperator(new OperatorBinaryFromSingleSeedInLabelMatN);
    //algorithm
    this->registerOperator(new OperatorVoronoiTesselationMatN);
    this->registerOperator(new OperatorGeodesicReconstructionMatN);
    this->registerOperator(new OperatorDynamicMatN);
    this->registerOperator(new OperatorDistanceMatN);
    this->registerOperator(new OperatorWatershedMatN);
    this->registerOperator(new OperatorMinimaMatN);
    this->registerOperator(new OperatorCluster2LabelMatN);
    this->registerOperator(new OperatorClusterMaxMatN);
    this->registerOperator(new OperatorAdamsBischofMatN);
    this->registerOperator(new OperatorHoleFillingMatN);

    this->registerOperator(new OperatorNonLinearAnistropicDiffusionMatN);
    this->registerOperator(new OperatorAllenCahnMatN);
    this->registerOperator(new OperatorCurvatureFromPhaseFieldMatN);
    this->registerOperator(new OperatorGetFieldFromMultiPhaseFieldMatN);
    //visualization
    //2D
    this->registerOperator(new OperatorColorGradationFromLabelMatN);
    this->registerOperator(new OperatorColorRandomFromLabelMatN);
    this->registerOperator(new OperatorColorAverageMatN);
    this->registerOperator(new OperatorColorForegroundMatN);

    //3D
    this->registerOperator(new OperatorAdditionOpenGl);
    this->registerOperator(new OperatorSetColorOpenGl);
    this->registerOperator(new OperatorCubeOpenGl);
    this->registerOperator(new OperatorPlaneOpenGl);
    this->registerOperator(new OperatorGraphOpenGl);
    this->registerOperator(new OperatorLatticeSurfaceOpenGl);
    this->registerOperator(new OperatorLineOpenGl);
    //this->registerOperator(new OperatorMarchingCubeBinaryImageGridOpenGl);
    this->registerOperator(new OperatorMarchingCubeColorImageGridOpenGl);
    //    this->registerOperator(new OperatorMarchingCubePhaseFieldMatNOpenGl);
    //    this->registerOperator(new OperatorMarchingSurfaceContactOpenGl);
    this->registerOperator(new OperatorTransparencyOpenGl);
    this->registerOperator(new OperatorAmbientOpenGl);
    this->registerOperator(new OperatorDiffuseOpenGl);

    //Geometry
    this->registerOperator(new OperatorCropMatN);
    this->registerOperator(new OperatorInsertImageMatN);
    this->registerOperator(new OperatorScaleMatN);
    //    this->registerOperator(new OperatorScaleHintSizeMatN);
    this->registerOperator(new OperatorTranslationMatN);
    this->registerOperator(new OperatorRotationMatN);
    this->registerOperator(new OperatorBlankMatN);
    this->registerOperator(new OperatorWhiteFaceMatN);

    //GermGrain
    this->registerOperator(new OperatorHardCoreGermGrain);
    this->registerOperator(new OperatorMinOverlapGermGrain);
    this->registerOperator(new OperatorRandomNonUniformPointGermGrain);
    this->registerOperator(new OperatorRandomUniformPointGermGrain);
    //Grain
    this->registerOperator(new OperatorBoxGermGrain);
    this->registerOperator(new OperatorCylinderGermGrain);
    this->registerOperator(new OperatorEllipsoidGermGrain);


    this->registerOperator(new OperatorPolyhedraGermGrain);
    this->registerOperator(new OperatorRhombohedronGermGrain);
    this->registerOperator(new OperatorSphereGermGrain);
    //    this->registerOperator(new OperatorGrainFromImageGermGrain);
    //Lattice
    this->registerOperator(new OperatorContinuousToLatticeGermGrain);
    //Model
    this->registerOperator(new OperatorBooleanGermGrain);
    this->registerOperator(new OperatorDeadLeaveGermGrain);
    this->registerOperator(new OperatorTransparencyGermGrain);
    this->registerOperator(new OperatorColorFromImageGermGrain);
    this->registerOperator(new OperatorRandomBlackOrWhiteGermGrain);
    this->registerOperator(new OperatorRandomColorGermGrain);

    //Art
    this->registerOperator(new OperatorRandomWalkGermGrain);

    //Geometrical constraint
    this->registerOperator(new OperatorGaussianRandomFieldMatN);
    this->registerOperator(new OperatorRandomStructureMatN);
    this->registerOperator(new OperatorAnnealingSimulatedMatN);

    //Analysis
    //Morphology

    //
    this->registerOperator(new OperatorVERPointPorosityMatN);
    this->registerOperator(new OperatorVERPointHistogramMatN);
    //Scalar
    this->registerOperator(new OperatorAreaMatN);
    this->registerOperator(new OperatorHistogramMatN);
    this->registerOperator(new OperatorMaxValueMatN);
    this->registerOperator(new OperatorMinValueMatN);
    this->registerOperator(new OperatorPerimeterMatN);
    //Statistic
    this->registerOperator(new OperatorChordMatN);
    this->registerOperator(new OperatorCorrelationMatN);
    this->registerOperator(new OperatorCorrelationGreyLevelMatN);
    this->registerOperator(new OperatorCorrelationDirectionByFFTMatN);
    this->registerOperator(new OperatorFractalBoxMatN);
    this->registerOperator(new OperatorMatheronGranulometryMatN);
    this->registerOperator(new OperatorLDistanceMatN);
    //Label
    this->registerOperator(new OperatorRepartitionAreaLabelMatN);
    this->registerOperator(new OperatorRepartitionPerimeterLabelMatN);
    this->registerOperator(new OperatorRepartitionPerimeterContactBetweenLabelMatN);
    this->registerOperator(new OperatorRepartitionFeretDiameterLabelMatN);

    //Scalar
    this->registerOperator(new OperatorPercolationMatN);
    this->registerOperator(new OperatorPercolationErosionMatN);
    this->registerOperator(new OperatorPercolationOpeningMatN);
    this->registerOperator(new OperatorEulerPoincareMatN);
    this->registerOperator(new OperatorGeometricalTortuosityMatN);


    //Skeleton
    this->registerOperator(new OperatorMedialAxisMatN);
    this->registerOperator(new OperatorThinningAtConstantTopologyMatN);
    this->registerOperator(new OperatorVertexAndEdgeFromSkeletonMatN);
    this->registerOperator(new OperatorLinkEdgeVertexMatN);

    this->registerOperator(new OperatorpermeabilityMatN);
    this->registerOperator(new OperatorDiffusionSelfCoefficientMatN);


    this->registerOperator(new OperatorFFTMatN);
    this->registerOperator(new OperatorFFTInverseMatN);

    //Input/Output
    this->registerOperator(new OperatorLoadDistribution);
    this->registerOperator(new OperatorSaveDistribution);
    //Distribution
    this->registerOperator(new OperatorUniformRealDistribution);
    this->registerOperator(new OperatorExponentielDistribution);
    this->registerOperator(new OperatorNormalDistribution);
    this->registerOperator(new OperatorDiracDistribution);
    this->registerOperator(new OperatorRegularExpressionDistribution);
    this->registerOperator(new OperatorFromMatrixDistribution);
    //    this->registerOperator(new OperatorBoxDistribution);
    //Discrete

    this->registerOperator(new OperatorUniformIntDistribution);
    this->registerOperator(new OperatorPoissonDistribution);
    this->registerOperator(new OperatorBinomialDistribution);
    this->registerOperator(new OperatorPencilDistribution);
    //Arithmetic
    this->registerOperator(new OperatorAddDistribution);
    this->registerOperator(new OperatorCompositionDistribution);
    this->registerOperator(new OperatorDivDistribution);
    this->registerOperator(new OperatorInverseDistribution);
    this->registerOperator(new OperatorMultDistribution);
    this->registerOperator(new OperatorOppositeDistribution);
    this->registerOperator(new OperatorSubDistribution);
    this->registerOperator(new OperatorMaxDistribution);
    //Operator
    this->registerOperator(new OperatorfofxDistribution);
    this->registerOperator(new OperatorArgMaxDistribution);
    this->registerOperator(new OperatorArgMinDistribution);
    this->registerOperator(new OperatorMaxValueDistribution);
    this->registerOperator(new OperatorMinValueDistribution);
    this->registerOperator(new OperatorMomentDistribution);
    this->registerOperator(new OperatorDistanceDistribution);
    this->registerOperator(new OperatorRandomVariableDistribution);
    this->registerOperator(new OperatorIntegralDistribution);
    this->registerOperator(new OperatorDerivateDistribution);
    this->registerOperator(new OperatorComputedStaticticsDistribution);

    //Convert
    this->registerOperator(new OperatorConvertMatrixDistribution);
    this->registerOperator(new OperatorConvertProbabilityDistributionDistribution);
    this->registerOperator(new OperatorConvertCumulativeDistribution);
    this->registerOperator(new OperatorConvertStepFunctionDistribution);

    this->registerOperator(new OperatorIntegralDistributionMultiVariate);
    this->registerOperator(new OperatorArgMaxDistributionMultiVariate);
    this->registerOperator(new OperatorArgMinDistributionMultiVariate);
    this->registerOperator(new OperatorMaxValueDistributionMultiVariate);
    this->registerOperator(new OperatorMinValueDistributionMultiVariate);
        this->registerOperator(new OperatorMomentDistributionMultiVariate);
    this->registerOperator(new OperatorConvertProbabilityDistributionMultiVariateDistributionMultiVariate);


    this->registerOperator(new OperatorSaveDistributionMultiVariate);
    this->registerOperator(new OperatorLoadDistributionMultiVariate);

    this->registerOperator(new OperatorNormalDistributionMultiVariate);
    this->registerOperator(new OperatorExpressionDistributionMultiVariate);
    this->registerOperator(new OperatorCoupledDistributionMultiVariate);
    this->registerOperator(new OperatorIndependantDistributionMultiVariate);
    this->registerOperator(new OperatorFromDistributionDistributionMultiVariate);

    //Arithmetic
    this->registerOperator(new OperatorAddDistributionMultiVariate);
    this->registerOperator(new OperatorCompositionDistributionMultiVariate);
    this->registerOperator(new OperatorDivDistributionMultiVariate);
    this->registerOperator(new OperatorInverseDistributionMultiVariate);
    this->registerOperator(new OperatorMultDistributionMultiVariate);
    this->registerOperator(new OperatorOppositeDistributionMultiVariate);
    this->registerOperator(new OperatorSubDistributionMultiVariate);
    this->registerOperator(new OperatorMaxDistributionMultiVariate);

    this->registerOperator(new OperatorRandomVariableDistributionMultiVariate);
    this->registerOperator(new OperatorfofxDistributionMultiVariate);


    //Linear Algebra
    this->registerOperator(new OperatorAddMatrix);
    this->registerOperator(new OperatorSubMatrix);
    this->registerOperator(new OperatorMultMatrix);
    this->registerOperator(new OperatorMultMatrixVectorMatrix);
    this->registerOperator(new OperatorMultMatrixScalarMatrix);

    this->registerOperator(new OperatorAddVector);
    this->registerOperator(new OperatorAddScalarVector);
    this->registerOperator(new OperatorSubVector);
    this->registerOperator(new OperatorMultVector);
    this->registerOperator(new OperatorMultEachTermVector);
    this->registerOperator(new OperatorMultVectorScalarVector);
    this->registerOperator(new OperatorMaxVector);
    this->registerOperator(new OperatorMinVector);


    this->registerOperator(new OperatorTransposeMatrixx);
    this->registerOperator(new OperatorInverseMatrix);
    this->registerOperator(new OperatorIdentityMatrix);
    this->registerOperator(new OperatorTraceMatrix);
    this->registerOperator(new OperatorDeterminantMatrix);
    this->registerOperator(new OperatorOrthogonalMatrix);
    this->registerOperator(new OperatorEigenValueMatrix);
    this->registerOperator(new OperatorEigenVectorMatrix);
    this->registerOperator(new OperatorBlankMatrix);
    this->registerOperator(new OperatorBlankPoint);
    this->registerOperator(new OperatorBlank2DPoint);
    this->registerOperator(new OperatorBlank3DPoint);
    this->registerOperator(new OperatorLoadMatrix);
    this->registerOperator(new OperatorSaveMatrix);
    this->registerOperator(new OperatorLoadPoint);
    this->registerOperator(new OperatorSavePoint);

    this->registerOperator(new OperatorConvertToTableMatrix);
    this->registerOperator(new OperatorConvertFromTableMatrix);

    this->registerOperator(new OperatorConvertToMatNMatrix);
    this->registerOperator(new OperatorConvertFromMatNMatrix);
    this->registerOperator(new OperatorSetMatrix);
    this->registerOperator(new OperatorGetMatrix);
    this->registerOperator(new OperatorSetRawMatrix);
    this->registerOperator(new OperatorSetColMatrix);
    this->registerOperator(new OperatorGetColMatrix);
    this->registerOperator(new OperatorGetRawMatrix);
    this->registerOperator(new OperatorResizeMatrix);
    this->registerOperator(new OperatorSizeMatrix);
    this->registerOperator(new OperatorSetPoint);
    this->registerOperator(new OperatorGetPoint);
    this->registerOperator(new OperatorResizePoint);
    this->registerOperator(new OperatorSizePoint);
    this->registerOperator(new OperatorPushBackPoint);
    this->registerOperator(new OperatorMultCoordinatePoint);
    //    this->registerOperator(new OperatorfofXColumnMatrix);
    this->registerOperator(new OperatorGenerate2DRotationMatrix);
    this->registerOperator(new OperatorGenerate3DRotationMatrix);



    this->registerOperator(new OperatorFromMatrixPlot);
    this->registerOperator(new OperatorFromDistributionPlot);

    this->registerOperator(new OperatorFromTablePlot);
    this->registerOperator(new OperatorAddGraphPlot);
    this->registerOperator(new OperatorBlankPlot);
    this->registerOperator(new OperatorFromMatrixPlot);
    this->registerOperator(new OperatorPopPointPlot);
    this->registerOperator(new OperatorPushPointPlot);
    this->registerOperator(new OperatorSetColorPlot);
    this->registerOperator(new OperatorSetLegendPlot);
    this->registerOperator(new OperatorSetTitlePlot);
    this->registerOperator(new OperatorSetWidthPlot);
    this->registerOperator(new OperatorSetAlphaPlot);
    this->registerOperator(new OperatorSetBrushColorPlot);
    this->registerOperator(new OperatorSetXAxisPlot);
    this->registerOperator(new OperatorSetYAxisPlot);
    this->registerOperator(new OperatorSetLogXAxisPlot);
    this->registerOperator(new OperatorSetLogYAxisPlot);


}

void PopulationDictionnary::collectControl(){
    this->registerControl(new ControlEditorMatN3D);
    this->registerControl(new ControlEditorMatN2D);
    this->registerControl(new ControlMarkerMatN);
    this->registerControl(new ControlMarker3DMatN);
    this->registerControl(new ControlEditorPoint);
    this->registerControl(new ControlEditorMatrix);
    this->registerControl(new ControlViewOpenGL);
    this->registerControl(new ControlViewMatN);
    this->registerControl(new ControlView3DMatN);
    this->registerControl(new ControlViewLabelMatN);
    this->registerControl(new ControlView3DMatNLabel);
    this->registerControl(new ControlViewMatrix);
    this->registerControl(new ControlViewPoint);
}

