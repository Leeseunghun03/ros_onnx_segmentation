#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <ros_onnx_segmentation/onnx_processor.h>
#include <ros_onnx_segmentation/yolo_processor.h>

#include <chrono>

using std::placeholders::_1;

const uint32_t kDefaultTensorWidth = 640;
const uint32_t kDefaultTensorHeight = 640;

using convert_t = std::codecvt_utf8<wchar_t>;
std::wstring_convert<convert_t, wchar_t> strconverter;
static std::wstring to_wstring(std::string str)
{
    return strconverter.from_bytes(str);
}

OnnxProcessor::OnnxProcessor() : _confidence(0.70f), _debug(false), _normalize(false), _process(ImageProcessing::Scale)
{
}

bool OnnxProcessor::init(rclcpp::Node::SharedPtr &node)
{
    _node = node;

    _node->get_parameter("confidence", _confidence);
    _node->get_parameter("debug", _debug);
    _node->get_parameter("link_name", _linkName);
    _fake = false;

    int temp = 0;
    if (_node->get_parameter("tensor_width", temp) && temp > 0)
    {
        _tensorWidth = (uint)temp;
    }
    else
    {
        _tensorWidth = kDefaultTensorWidth;
    }

    temp = 0;
    if (_node->get_parameter("tensor_height", temp) && temp > 0)
    {
        _tensorHeight = (uint)temp;
    }
    else
    {
        _tensorHeight = kDefaultTensorHeight;
    }

    if (!_node->get_parameter("onnx_model_path", _onnxModel) ||
        _onnxModel.empty())
    {
        RCLCPP_ERROR(_node->get_logger(), "Onnx: onnx_model_path parameter has not been set.");
        return false;
    }

    if (_node->get_parameter("calibration", _calibration))
    {
        try
        {
            cv::FileStorage fs(_calibration, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
            fs["camera_matrix"] >> _camera_matrix;
            fs["distortion_coefficients"] >> _dist_coeffs;
        }
        catch (std::exception &e)
        {
            RCLCPP_ERROR(_node->get_logger(), "Failed to read the calibration file, continuing without calibration.\n%s", e.what());
            // no calibration for you.
            _calibration = "";
        }
    }

    std::string imageProcessingType;
    if (_node->get_parameter("image_processing", imageProcessingType))
    {
        if (imageProcessingType == "crop")
        {
            _process = Crop;
        }
        else if (imageProcessingType == "scale")
        {
            _process = Scale;
        }
        else if (imageProcessingType == "resize")
        {
            _process = Resize;
        }
        else
        {
            RCLCPP_ERROR(_node->get_logger(), "Onnx: unknown image processing type: %s", imageProcessingType.c_str());
            // default;
        }
    }

    // Generate onnx session
    //*************************************************************************
    // initialize  enviroment...one enviroment per process
    // enviroment maintains thread pools and other state info
    _env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");

    // initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
    auto modelFullPath = ::to_wstring(_onnxModel).c_str();
#else
    auto modelFullPath = _onnxModel.c_str();
#endif

    _session = std::make_shared<Ort::Session>(*_env, modelFullPath, session_options);
    _allocator = std::make_shared<Ort::AllocatorWithDefaultOptions>();
    DumpParameters();

    // Set up publishers and subscribers
    std::string image_topic_ = "camera/image_raw";
    std::string image_pub_topic_ = "image_debug_raw";

    _node->get_parameter("image_topic", image_topic_);
    _node->get_parameter("image_debug_topic", image_pub_topic_);

    image_pub_ = _node->create_publisher<sensor_msgs::msg::Image>(image_pub_topic_, 10);
    subscription_ = _node->create_subscription<sensor_msgs::msg::Image>(
        image_topic_, 10, std::bind(&OnnxProcessor::ProcessImage, this, _1));

    std::cout << "ONNX Processor successfully initialized." << std::endl;

    return true;
}

void OnnxProcessor::ProcessImage(const sensor_msgs::msg::Image::SharedPtr msg)
{
    if (_session == nullptr)
    {
        return;
    }

    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat raw_image = cv_ptr->image.clone();

    // Resize image
    cv::Size mlSize(_tensorWidth, _tensorHeight);
    cv::Mat image_resized;
    cv::Size s = cv_ptr->image.size();
    float aspectRatio = (float)s.width / (float)s.height;
    if (s.width <= 0 || s.height <= 0)
    {
        RCLCPP_ERROR(_node->get_logger(), "ONNX: irrational image size received; one dimention zero or less");
        return;
    }

    if (_process == Crop && (uint)s.width > _tensorWidth && (uint)s.height > _tensorHeight)
    {
        cv::Rect ROI((s.width - _tensorWidth) / 2, (s.height - _tensorHeight) / 2, _tensorWidth, _tensorHeight);
        image_resized = cv_ptr->image(ROI);
    }
    else if (_process == Resize)
    {
        cv::resize(cv_ptr->image, image_resized, mlSize, 0, 0, cv::INTER_CUBIC);
    }
    else // Scale
    {
        cv::Size downsampleSize;

        if (aspectRatio > 1.0f)
        {
            downsampleSize.height = mlSize.height;
            downsampleSize.width = (int)(mlSize.height * aspectRatio);
        }
        else
        {
            downsampleSize.width = mlSize.width;
            downsampleSize.height = (int)(mlSize.width * aspectRatio);
        }

        cv::resize(cv_ptr->image, image_resized, downsampleSize, 0, 0, cv::INTER_CUBIC);

        cv::Rect ROI((downsampleSize.width - _tensorWidth) / 2, (downsampleSize.height - _tensorHeight) / 2, _tensorWidth, _tensorHeight);
        image_resized = image_resized(ROI);
    }

    // Convert to RGB
    cv::Mat rgb_image;
    cv::cvtColor(image_resized, rgb_image, cv::COLOR_BGR2RGB);

    // Set the image to 32-bit floating point values for tensorization.
    cv::Mat image_32_bit;
    rgb_image.convertTo(image_32_bit, CV_32F);

    if (_normalize)
    {
        cv::normalize(image_32_bit, image_32_bit, 0.0f, 1.0f, cv::NORM_MINMAX);
    }

    cv::Mat channels[3];
    cv::split(image_32_bit, channels);

    size_t input_tensor_size = _tensorHeight * _tensorWidth * 3;

    std::vector<float> input_tensor_values(input_tensor_size);

    memmove(&input_tensor_values[0], (float *)channels[0].data, _tensorWidth * _tensorHeight * sizeof(float));
    memmove(&input_tensor_values[_tensorWidth * _tensorHeight], (float *)channels[1].data, _tensorWidth * _tensorHeight * sizeof(float));
    memmove(&input_tensor_values[2 * _tensorWidth * _tensorHeight], (float *)channels[2].data, _tensorWidth * _tensorHeight * sizeof(float));

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::TypeInfo type_info = _session->GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_node_dims = {1, 3, 640, 640};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size());

    std::vector<const char *> input_node_names = _inName;
    std::vector<const char *> output_node_names = _outName;

    auto start = std::chrono::high_resolution_clock::now();
    auto outputs = _session->Run(Ort::RunOptions{nullptr},
                                                   input_node_names.data(), &input_tensor, 1, output_node_names.data(), output_node_names.size());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "time: " << duration << " millis." << std::endl;

    float* all_data = outputs[0].GetTensorMutableData<float>();
	auto data_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	cv::Mat output0 = cv::Mat(cv::Size((int)data_shape[2], (int)data_shape[1]), CV_32F, all_data).t();
	auto mask_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
	std::vector<int> mask_sz = { 1,(int)mask_shape[1],(int)mask_shape[2],(int)mask_shape[3] };
	cv::Mat output1 = cv::Mat(mask_sz, CV_32F, outputs[1].GetTensorMutableData<float>());

    ProcessOutput(output0, output1, raw_image);
}

void OnnxProcessor::DumpParameters()
{
    //*************************************************************************
    // print model input layer (node names, types, shape etc.)

    // print number of model input nodes
    size_t num_input_nodes = _session->GetInputCount();
    std::vector<const char *> input_node_names;
    input_node_names.resize(num_input_nodes);
    std::vector<int64_t> input_node_dims; // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                          // Otherwise need vector<vector<>>

    printf("Number of inputs = %zu\n", num_input_nodes);

    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++)
    {
        // print input node names
        auto input_name = _session->GetInputNameAllocated(i, *_allocator);
        printf("Input %d : name=%s\n", i, input_name.get());
        input_node_names[i] = input_name.get();

        // print input node types
        Ort::TypeInfo type_info = _session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        for (int j = 0; j < input_node_dims.size(); j++)
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
    }

    //*************************************************************************
    // print model output layer (node names, types, shape etc.)
    size_t num_output_nodes = _session->GetOutputCount();
    std::vector<const char *> output_node_names;
    output_node_names.resize(num_output_nodes);
    std::vector<int64_t> output_node_dims; // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                           // Otherwise need vector<vector<>>

    printf("Number of outputs = %zu\n", num_output_nodes);

    // iterate over all output nodes
    for (int i = 0; i < num_output_nodes; i++)
    {
        // print output node names
        auto output_name = _session->GetOutputNameAllocated(i, *_allocator);
        printf("Output %d : name=%s\n", i, output_name.get());
        output_node_names[i] = output_name.get();

        // print input node types
        Ort::TypeInfo type_info = _session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Output %d : type=%d\n", i, type);

        // print input shapes/dims
        output_node_dims = tensor_info.GetShape();
        printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
        for (int j = 0; j < output_node_dims.size(); j++)
            printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
    }
}

bool OnnxTracker::init(rclcpp::Node::SharedPtr &node)
{
    // Declare nodes parameters
    node->declare_parameter<std::string>("tracker_type", "yolo");
    node->declare_parameter<std::string>("onnx_model_path", "");
    node->declare_parameter<double>("confidence", 0.5);
    node->declare_parameter<uint16_t>("tensor_width", 640);
    node->declare_parameter<uint16_t>("tensor_height", 640);
    node->declare_parameter<bool>("debug", false);
    node->declare_parameter<std::string>("image_processing", "resize");
    node->declare_parameter<std::string>("label", "person");
    node->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    node->declare_parameter<std::string>("image_debug_topic", "image_debug_raw");

    _processor = std::make_shared<yolo::YoloProcessor>();

    return _processor->init(node);
}