#include <ros_onnx_segmentation/onnx_processor.h>

#include <array>

#ifdef _WIN32
#include <codecvt>
#include <locale>
#endif

using std::placeholders::_1;

bool OnnxProcessor::init(rclcpp::Node::SharedPtr &node)
{
    _node = node;

    auto model_path = _node->declare_parameter<std::string>("onnx_model_path", "");
    auto image_topic = _node->declare_parameter<std::string>("image_topic", "camera/image_raw");
    auto processing = _node->declare_parameter<std::string>("image_processing", "letterbox");
    auto num_threads = _node->declare_parameter<int>("num_threads", 1);
    auto tensor_width = _node->declare_parameter<int>("tensor_width", 640);
    auto tensor_height = _node->declare_parameter<int>("tensor_height", 640);
    _debug = _node->declare_parameter<bool>("debug", false);

    if (model_path.empty())
    {
        RCLCPP_ERROR(_node->get_logger(), "onnx_model_path is not set");
        return false;
    }

    if (processing == "resize")
    {
        _letterbox = false;
    }
    else if (processing != "letterbox")
    {
        RCLCPP_WARN(_node->get_logger(), "Unknown image_processing '%s', using letterbox", processing.c_str());
    }

    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(num_threads);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    _env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ros_onnx_segmentation");
    try
    {
#ifdef _WIN32
        std::wstring wpath = std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(model_path);
        _session = std::make_unique<Ort::Session>(*_env, wpath.c_str(), options);
#else
        _session = std::make_unique<Ort::Session>(*_env, model_path.c_str(), options);
#endif
    }
    catch (const Ort::Exception &e)
    {
        RCLCPP_ERROR(_node->get_logger(), "Failed to load %s: %s", model_path.c_str(), e.what());
        return false;
    }

    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < _session->GetInputCount(); ++i)
        _inputNames.emplace_back(_session->GetInputNameAllocated(i, allocator).get());
    for (size_t i = 0; i < _session->GetOutputCount(); ++i)
        _outputNames.emplace_back(_session->GetOutputNameAllocated(i, allocator).get());
    for (const auto &name : _inputNames)
        _inputNamePtrs.push_back(name.c_str());
    for (const auto &name : _outputNames)
        _outputNamePtrs.push_back(name.c_str());

    // A static input shape in the model overrides tensor_width/height.
    auto shape = _session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    bool is_static = shape.size() == 4 && shape[2] > 0 && shape[3] > 0;
    _tensorHeight = is_static ? static_cast<int>(shape[2]) : tensor_height;
    _tensorWidth = is_static ? static_cast<int>(shape[3]) : tensor_width;

    RCLCPP_INFO(_node->get_logger(), "Loaded %s (input %dx%d)", model_path.c_str(), _tensorWidth, _tensorHeight);

    _imageSub = _node->create_subscription<sensor_msgs::msg::Image>(
        image_topic, rclcpp::SensorDataQoS(), std::bind(&OnnxProcessor::ProcessImage, this, _1));

    return true;
}

cv::Mat OnnxProcessor::Preprocess(const cv::Mat &image, ImageInfo &info) const
{
    info.raw_size = image.size();
    cv::Mat out;

    if (!_letterbox)
    {
        info.scale_x = static_cast<double>(_tensorWidth) / image.cols;
        info.scale_y = static_cast<double>(_tensorHeight) / image.rows;
        info.pad_x = info.pad_y = 0.0;
        cv::resize(image, out, cv::Size(_tensorWidth, _tensorHeight));
        return out;
    }

    double r = std::min(static_cast<double>(_tensorWidth) / image.cols,
                        static_cast<double>(_tensorHeight) / image.rows);
    int w = static_cast<int>(std::round(image.cols * r));
    int h = static_cast<int>(std::round(image.rows * r));
    int pad_x = (_tensorWidth - w) / 2;
    int pad_y = (_tensorHeight - h) / 2;

    info.scale_x = info.scale_y = r;
    info.pad_x = pad_x;
    info.pad_y = pad_y;

    out = cv::Mat(_tensorHeight, _tensorWidth, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(image, out(cv::Rect(pad_x, pad_y, w, h)), cv::Size(w, h));
    return out;
}

void OnnxProcessor::ProcessImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (const cv_bridge::Exception &e)
    {
        RCLCPP_ERROR_THROTTLE(_node->get_logger(), *_node->get_clock(), 5000, "cv_bridge: %s", e.what());
        return;
    }

    const cv::Mat &image = cv_ptr->image;
    if (image.empty())
        return;

    ImageInfo info;
    cv::Mat input = Preprocess(image, info);
    cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);

    std::array<int64_t, 4> shape = {1, 3, _tensorHeight, _tensorWidth};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memory_info, blob.ptr<float>(), blob.total(), shape.data(), shape.size());

    std::vector<Ort::Value> outputs;
    try
    {
        outputs = _session->Run(Ort::RunOptions{nullptr}, _inputNamePtrs.data(), &tensor, 1,
                                _outputNamePtrs.data(), _outputNamePtrs.size());
    }
    catch (const Ort::Exception &e)
    {
        RCLCPP_ERROR_THROTTLE(_node->get_logger(), *_node->get_clock(), 5000, "Inference failed: %s", e.what());
        return;
    }

    ProcessOutput(outputs, info, image, msg->header);
}
