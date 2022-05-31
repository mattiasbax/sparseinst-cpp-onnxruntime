#include <iostream>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include <chrono>
#include <fstream>
#include <iostream>

namespace
{

std::vector<float> loadImage( const std::string& filename, cv::Size size )
{
    cv::Mat image = cv::imread( filename );      // read image
    cvtColor( image, image, cv::COLOR_BGR2RGB ); // convert from bgr  to rgb
    cv::resize( image, image, size );            // resize to network image size
    image = image.reshape( 1, 1 );               // flatten to 1D
    std::vector<float> imageData;
    image.convertTo( imageData, CV_32FC1 ); // convert to float
    return imageData;
}

struct ModelParameters {
    size_t numInputNodes;
    std::vector<const char*> inputNodeNames;
    std::vector<std::vector<int64_t>> inputNodesDimensions;
    size_t numOutputNodes;
    std::vector<const char*> outputNodeNames;
    std::vector<std::vector<int64_t>> outputNodesDimensions;
};

ModelParameters getModelParameters( const Ort::Session& session, bool printInfo = false )
{
    ModelParameters mp;

    Ort::AllocatorWithDefaultOptions allocator; // TODO: Free memory
    mp.numInputNodes = session.GetInputCount( );
    for ( size_t idx = 0; idx < mp.numInputNodes; ++idx ) {
        mp.inputNodeNames.push_back( session.GetInputName( idx, allocator ) );

        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo( idx );
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo( );
        mp.inputNodesDimensions.emplace_back( inputTensorInfo.GetShape( ) );
    }

    mp.numOutputNodes = session.GetOutputCount( );
    for ( size_t idx = 0; idx < mp.numOutputNodes; ++idx ) {
        mp.outputNodeNames.push_back( session.GetOutputName( idx, allocator ) );

        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo( idx );
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo( );
        mp.outputNodesDimensions.emplace_back( outputTensorInfo.GetShape( ) );
    }

    if ( !printInfo )
        return mp;

    std::cout << "Number of Input Nodes: " << mp.numInputNodes << std::endl;
    for ( size_t idx = 0; idx < mp.numInputNodes; ++idx ) {
        std::cout << "Input Name " << idx << ": " << mp.inputNodeNames[ idx ] << std::endl;
        std::cout << "Input " << idx << " Dimensions: ( ";
        for ( const auto& i : mp.inputNodesDimensions[ idx ] ) {
            std::cout << i << " ";
        }
        std::cout << ")" << std::endl;
    }

    std::cout << "Number of Output Nodes: " << mp.numOutputNodes << std::endl;
    for ( size_t idx = 0; idx < mp.numOutputNodes; ++idx ) {
        std::cout << "Output Name " << idx << ": " << mp.outputNodeNames[ idx ] << std::endl;
        std::cout << "Output " << idx << " Dimensions: ( ";
        for ( const auto& i : mp.outputNodesDimensions[ idx ] ) {
            std::cout << i << " ";
        }
        std::cout << ")" << std::endl;
    }

    return mp;
}

} // namespace

int main( )
{
    ///////////// Create session
    constexpr bool useCuda = true;
    const std::string instanceName = "SparseInst";

    const std::string imageFilepath = "C:\\Users\\mattias.backstrom\\models\\inframe.jpg";

    Ort::Env env( OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str( ) );
    Ort::SessionOptions sessionOptions;
    // sessionOptions.SetIntraOpNumThreads( 1 );
    // sessionOptions.SetGraphOptimizationLevel( GraphOptimizationLevel:: );
    OrtCUDAProviderOptions cudaOptions;
    cudaOptions.device_id = 0;
    sessionOptions.AppendExecutionProvider_CUDA( cudaOptions );
    Ort::Session session( env, L"C:\\Users\\mattias.backstrom\\models\\sparse_inst_r50_giam_aug_2b7d68.onnx", sessionOptions );

    ModelParameters mp = getModelParameters( session, true );

    ///////////// Prepare input
    constexpr int numChannels = 3;
    const cv::Size inputImageSize = cv::Size( 640, 640 );
    std::vector<float> inputImageData = loadImage( imageFilepath, inputImageSize );
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu( OrtDeviceAllocator, OrtMemTypeDefault );
    const std::array<int64_t, 4> inputShape = { 1, inputImageSize.width, inputImageSize.height, numChannels };
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>( memoryInfo, inputImageData.data( ), inputImageData.size( ), inputShape.data( ), inputShape.size( ) );

    ///////////// Run inference

    constexpr int numberOfInferences = 1;
    const float* inputData = inputTensor.GetTensorData<float>( );

    std::vector<Ort::Value> outputTensors;
    outputTensors = session.Run( Ort::RunOptions{ nullptr }, mp.inputNodeNames.data( ), &inputTensor, mp.numInputNodes, mp.outputNodeNames.data( ), mp.numOutputNodes );

    auto start = std::chrono::system_clock::now( );
    for ( int i = 0; i < numberOfInferences; ++i ) {
        outputTensors = session.Run( Ort::RunOptions{ nullptr }, mp.inputNodeNames.data( ), &inputTensor, mp.numInputNodes, mp.outputNodeNames.data( ), mp.numOutputNodes );
    }
    auto end = std::chrono::system_clock::now( );
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end - start );
    std::cout << "Average inference time [ms]:" << elapsed.count( ) / numberOfInferences << '\n';

    ///////////// Postprocess data

    const Ort::Value& masks = outputTensors[ 0 ];
    const Ort::Value& scores = outputTensors[ 1 ];
    const Ort::Value& labels = outputTensors[ 2 ];

    const bool* masksData = masks.GetTensorData<bool>( );
    const float* scoresData = scores.GetTensorData<float>( );
    const int64_t* labelsData = labels.GetTensorData<int64_t>( );

    cv::Mat underlay = cv::imread( imageFilepath );
    cv::Mat green( underlay.size( ), CV_8UC3, cv::Scalar( 0., 75., 0. ) );

    const int64_t maxNumberOfMasks = scores.GetTensorTypeAndShapeInfo( ).GetShape( ).at( 1 );
    const float confidenceThreshold = 0.6f;
    const std::vector<int> classesToDetect = { 0 };
    std::vector<cv::Mat> filteredMasks;
    for ( int64_t idx = 0; idx < maxNumberOfMasks; ++idx ) {
        if ( ( scoresData[ idx ] ) < confidenceThreshold || std::find( classesToDetect.begin( ), classesToDetect.end( ), labelsData[ idx ] ) == classesToDetect.end( ) )
            continue;
        std::cout << "Detecteced label " << labelsData[ idx ] << " with confidence: " << scoresData[ idx ] << std::endl;
        filteredMasks.push_back( cv::Mat( inputImageSize, CV_8UC1, (uchar*) masksData + inputImageSize.area( ) * idx ) );
        cv::Mat& mask = filteredMasks.back( );

        cv::resize( mask, mask, underlay.size( ) );
        cv::add( green, underlay, underlay, mask );

        std::vector<cv::Mat> contours;
        cv::Mat hierarchy;
        cv::findContours( mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE );
        cv::drawContours( underlay, contours, -1, cv::Scalar{ 0., 255., 0. }, 2, cv::LINE_8, hierarchy, 100 );
    }

    cv::imshow( "result", underlay );
    cv::waitKey( 0 );
    
    return 0;
}