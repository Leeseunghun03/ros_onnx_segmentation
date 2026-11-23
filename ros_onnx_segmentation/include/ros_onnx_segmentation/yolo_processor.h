#pragma once

#include <string>
#include <vector>

#include <ros_onnx_segmentation/onnx_processor.h>
#include <vision_msgs/msg/detection2_d_array.hpp>

namespace yolo
{
    struct Obj
    {
        int id = 0;
        float accu = 0.0f;
        cv::Rect bound;
        cv::Mat mask;
    };

    // YOLOv8 / YOLO11 segmentation output:
    // output0 [1, 4 + nc + nm, N], output1 [1, nm, mh, mw]
    class YoloProcessor : public OnnxProcessor
    {
    public:
        bool init(rclcpp::Node::SharedPtr &node) override;

    protected:
        void ProcessOutput(std::vector<Ort::Value> &outputs, const ImageInfo &info,
                           const cv::Mat &image, const std_msgs::msg::Header &header) override;

    private:
        std::vector<std::string> read_class_names() const;
        cv::Mat get_mask(const cv::Mat &coeffs, const cv::Mat &protos, const ImageInfo &info, const cv::Rect &bound) const;
        cv::Mat draw_result(const cv::Mat &image, std::vector<Obj> objs) const;

        std::vector<std::string> _classNames;
        std::vector<cv::Scalar> _colors;

        float _confThresh = 0.35f;
        float _iouThresh = 0.6f;
        float _maskLogit = 0.0f;

        size_t _detIndex = 0;
        size_t _protoIndex = 1;
        int _maskChannels = 32;
        int _protoWidth = 160;
        int _protoHeight = 160;

        rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr _detectionPub;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _maskPub;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _debugPub;
    };
}
