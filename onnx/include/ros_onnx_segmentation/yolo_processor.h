// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ros_onnx_segmentation/onnx_processor.h>

namespace yolo
{
    struct YoloBox
    {
    public:
        std::string label;
        float x, y, width, height, confidence;
    };

    struct Obj
    {
        int id = 0;
        float accu = 0.0;
        cv::Rect bound;
        cv::Mat mask;
    };

    struct ImageInfo
    {
        cv::Size raw_size;
        cv::Vec4d trans;
    };

    class YoloProcessor : public OnnxProcessor
    {
        std::string _label;

    public:
        YoloProcessor();
        virtual bool init(std::shared_ptr<rclcpp::Node> &node);

    protected:
        void get_mask(const cv::Mat& mask_info, const cv::Mat& mask_data, const ImageInfo& para, cv::Rect bound, cv::Mat& mast_out);
        virtual void ProcessOutput(const cv::Mat &output0, const cv::Mat &output1, const cv::Mat &raw_image);
        void draw_result(cv::Mat img, std::vector<Obj> &result, std::vector<cv::Scalar> color);

    private:
        std::vector<cv::Scalar> color;
    };
}