#include <ros_onnx_segmentation/yolo_processor.h>

#include <algorithm>
#include <cmath>
#include <regex>

namespace yolo
{
    bool YoloProcessor::init(rclcpp::Node::SharedPtr &node)
    {
        if (!OnnxProcessor::init(node))
            return false;

        _confThresh = static_cast<float>(_node->declare_parameter<double>("conf_thresh", 0.35));
        _iouThresh = static_cast<float>(_node->declare_parameter<double>("iou_thresh", 0.6));
        double mask_thresh = _node->declare_parameter<double>("mask_thresh", 0.3);
        auto debug_topic = _node->declare_parameter<std::string>("image_debug_topic", "image_debug_raw");
        _classNames = _node->declare_parameter<std::vector<std::string>>("class_names", std::vector<std::string>{});

        // Compare against the logit instead of applying sigmoid to every pixel.
        mask_thresh = std::min(std::max(mask_thresh, 1e-6), 1.0 - 1e-6);
        _maskLogit = static_cast<float>(std::log(mask_thresh / (1.0 - mask_thresh)));

        if (_outputNames.size() != 2)
        {
            RCLCPP_ERROR(_node->get_logger(), "Expected a segmentation model with 2 outputs, got %zu", _outputNames.size());
            return false;
        }

        std::vector<int64_t> shapes[2];
        for (size_t i = 0; i < 2; ++i)
        {
            shapes[i] = _session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            if (shapes[i].size() == 4)
                _protoIndex = i;
        }
        _detIndex = 1 - _protoIndex;

        const auto &det = shapes[_detIndex];
        const auto &proto = shapes[_protoIndex];
        if (det.size() != 3 || proto.size() != 4)
        {
            RCLCPP_ERROR(_node->get_logger(), "Unexpected output shapes");
            return false;
        }

        _maskChannels = static_cast<int>(proto[1]);
        _protoHeight = static_cast<int>(proto[2]);
        _protoWidth = static_cast<int>(proto[3]);
        int num_classes = static_cast<int>(det[1]) - 4 - _maskChannels;
        if (num_classes <= 0)
        {
            RCLCPP_ERROR(_node->get_logger(), "Could not infer the number of classes from the model");
            return false;
        }

        if (_classNames.empty())
            _classNames = read_class_names();
        if (static_cast<int>(_classNames.size()) != num_classes)
        {
            if (!_classNames.empty())
                RCLCPP_WARN(_node->get_logger(), "class_names has %zu entries but the model has %d classes",
                            _classNames.size(), num_classes);
            _classNames.resize(num_classes);
            for (int i = 0; i < num_classes; ++i)
            {
                if (_classNames[i].empty())
                    _classNames[i] = "class_" + std::to_string(i);
            }
        }

        // Golden-ratio hue steps keep neighboring instances distinguishable.
        cv::Mat hsv(1, 32, CV_8UC3);
        for (int i = 0; i < hsv.cols; ++i)
            hsv.at<cv::Vec3b>(0, i) = cv::Vec3b(static_cast<uchar>(std::fmod(i * 0.618034, 1.0) * 180), 220, 255);
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        for (int i = 0; i < bgr.cols; ++i)
            _colors.emplace_back(bgr.at<cv::Vec3b>(0, i));

        _detectionPub = _node->create_publisher<vision_msgs::msg::Detection2DArray>("detections", 10);
        _maskPub = _node->create_publisher<sensor_msgs::msg::Image>("instance_mask", 10);
        if (_debug)
            _debugPub = _node->create_publisher<sensor_msgs::msg::Image>(debug_topic, 10);

        RCLCPP_INFO(_node->get_logger(), "%d classes, mask prototypes %dx%dx%d",
                    num_classes, _maskChannels, _protoWidth, _protoHeight);
        return true;
    }

    std::vector<std::string> YoloProcessor::read_class_names() const
    {
        std::vector<std::string> names;
        Ort::AllocatorWithDefaultOptions allocator;
        auto metadata = _session->GetModelMetadata();
        auto value = metadata.LookupCustomMetadataMapAllocated("names", allocator);
        if (!value)
            return names;

        // Ultralytics stores a Python dict literal: {0: 'person', 1: 'bicycle', ...}
        std::string text = value.get();
        std::regex pattern(R"((\d+):\s*(['"])(.*?)\2)");
        for (auto it = std::sregex_iterator(text.begin(), text.end(), pattern); it != std::sregex_iterator(); ++it)
        {
            size_t id = std::stoul((*it)[1].str());
            if (id >= names.size())
                names.resize(id + 1);
            names[id] = (*it)[3].str();
        }
        return names;
    }

    void YoloProcessor::ProcessOutput(std::vector<Ort::Value> &outputs, const ImageInfo &info,
                                      const cv::Mat &image, const std_msgs::msg::Header &header)
    {
        auto det_shape = outputs[_detIndex].GetTensorTypeAndShapeInfo().GetShape();
        cv::Mat det = cv::Mat(static_cast<int>(det_shape[1]), static_cast<int>(det_shape[2]), CV_32F,
                              outputs[_detIndex].GetTensorMutableData<float>()).t();
        cv::Mat protos(_maskChannels, _protoWidth * _protoHeight, CV_32F,
                       outputs[_protoIndex].GetTensorMutableData<float>());

        const int num_classes = static_cast<int>(_classNames.size());
        const cv::Rect image_rect(cv::Point(0, 0), info.raw_size);

        std::vector<int> class_ids;
        std::vector<int> rows;
        std::vector<float> scores;
        std::vector<cv::Rect> boxes;
        std::vector<cv::Rect> nms_boxes;

        for (int r = 0; r < det.rows; ++r)
        {
            const float *p = det.ptr<float>(r);
            const float *best = std::max_element(p + 4, p + 4 + num_classes);
            if (*best <= _confThresh)
                continue;

            double cx = (p[0] - info.pad_x) / info.scale_x;
            double cy = (p[1] - info.pad_y) / info.scale_y;
            double w = p[2] / info.scale_x;
            double h = p[3] / info.scale_y;
            cv::Rect box(cvRound(cx - w / 2), cvRound(cy - h / 2), cvRound(w), cvRound(h));
            box &= image_rect;
            if (box.empty())
                continue;

            int id = static_cast<int>(best - (p + 4));
            class_ids.push_back(id);
            rows.push_back(r);
            scores.push_back(*best);
            boxes.push_back(box);
            // Shift boxes per class so NMS does not suppress across classes.
            int offset = id * (info.raw_size.width + info.raw_size.height);
            nms_boxes.push_back(box + cv::Point(offset, offset));
        }

        std::vector<int> keep;
        cv::dnn::NMSBoxes(nms_boxes, scores, _confThresh, _iouThresh, keep);

        vision_msgs::msg::Detection2DArray det_msg;
        det_msg.header = header;
        std::vector<Obj> objs;

        for (int idx : keep)
        {
            Obj obj;
            obj.id = class_ids[idx];
            obj.accu = scores[idx];
            obj.bound = boxes[idx];

            cv::Mat coeffs(1, _maskChannels, CV_32F, det.ptr<float>(rows[idx]) + 4 + num_classes);
            obj.mask = get_mask(coeffs, protos, info, obj.bound);

            vision_msgs::msg::Detection2D d;
            d.header = header;
            // Matches the pixel value of this instance in instance_mask.
            d.id = std::to_string(objs.size() + 1);
            d.bbox.center.position.x = obj.bound.x + obj.bound.width / 2.0;
            d.bbox.center.position.y = obj.bound.y + obj.bound.height / 2.0;
            d.bbox.size_x = obj.bound.width;
            d.bbox.size_y = obj.bound.height;

            vision_msgs::msg::ObjectHypothesisWithPose hyp;
            hyp.hypothesis.class_id = _classNames[obj.id];
            hyp.hypothesis.score = obj.accu;
            d.results.push_back(hyp);
            det_msg.detections.push_back(std::move(d));

            objs.push_back(std::move(obj));
        }

        _detectionPub->publish(det_msg);

        if (_maskPub->get_subscription_count() > 0)
        {
            // 0 is background. Higher-score instances are written first and keep overlapping pixels.
            cv::Mat ids = cv::Mat::zeros(info.raw_size, CV_16UC1);
            for (size_t i = 0; i < objs.size(); ++i)
            {
                cv::Mat roi = ids(objs[i].bound);
                roi.setTo(cv::Scalar(static_cast<double>(i + 1)), objs[i].mask & (roi == 0));
            }
            _maskPub->publish(*cv_bridge::CvImage(header, "mono16", ids).toImageMsg());
        }

        if (_debugPub && _debugPub->get_subscription_count() > 0)
        {
            auto msg = cv_bridge::CvImage(header, "bgr8", draw_result(image, objs)).toImageMsg();
            _debugPub->publish(*msg);
        }
    }

    cv::Mat YoloProcessor::get_mask(const cv::Mat &coeffs, const cv::Mat &protos, const ImageInfo &info,
                                    const cv::Rect &bound) const
    {
        cv::Mat logits = cv::Mat(coeffs * protos).reshape(1, _protoHeight);

        // Map each pixel center of the box back to prototype coordinates.
        const double sx = static_cast<double>(_protoWidth) / _tensorWidth;
        const double sy = static_cast<double>(_protoHeight) / _tensorHeight;
        cv::Matx23d m(info.scale_x * sx, 0.0, ((bound.x + 0.5) * info.scale_x + info.pad_x) * sx - 0.5,
                      0.0, info.scale_y * sy, ((bound.y + 0.5) * info.scale_y + info.pad_y) * sy - 0.5);

        cv::Mat warped;
        cv::warpAffine(logits, warped, m, bound.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                       cv::BORDER_REPLICATE);
        return warped > _maskLogit;
    }

    std::vector<size_t> YoloProcessor::assign_colors(const std::vector<Obj> &objs)
    {
        // Reuse the color of the best overlapping box of the same class in the previous frame,
        // so the overlay does not flicker. Only affects the debug image.
        std::vector<size_t> colors(objs.size(), _colors.size());
        std::vector<bool> taken(_prevBoxes.size(), false);
        std::vector<bool> used(_colors.size(), false);

        for (size_t i = 0; i < objs.size(); ++i)
        {
            double best_iou = 0.3;
            size_t match = _prevBoxes.size();
            for (size_t j = 0; j < _prevBoxes.size(); ++j)
            {
                if (taken[j] || _prevBoxes[j].id != objs[i].id)
                    continue;
                double inter = (objs[i].bound & _prevBoxes[j].bound).area();
                double iou = inter / (objs[i].bound.area() + _prevBoxes[j].bound.area() - inter);
                if (iou > best_iou)
                {
                    best_iou = iou;
                    match = j;
                }
            }
            if (match < _prevBoxes.size())
            {
                taken[match] = true;
                colors[i] = _prevBoxes[match].color;
                used[colors[i]] = true;
            }
        }

        for (size_t i = 0; i < objs.size(); ++i)
        {
            if (colors[i] < _colors.size())
                continue;
            for (size_t k = 0; k < _colors.size(); ++k)
            {
                size_t c = (_nextColor + k) % _colors.size();
                if (!used[c] || k + 1 == _colors.size())
                {
                    colors[i] = c;
                    break;
                }
            }
            used[colors[i]] = true;
            _nextColor = (colors[i] + 1) % _colors.size();
        }

        _prevBoxes.clear();
        for (size_t i = 0; i < objs.size(); ++i)
            _prevBoxes.push_back({objs[i].bound, objs[i].id, colors[i]});
        return colors;
    }

    cv::Mat YoloProcessor::draw_result(const cv::Mat &image, const std::vector<Obj> &objs)
    {
        std::vector<size_t> colors = assign_colors(objs);

        cv::Mat overlay = image.clone();
        for (size_t i = 0; i < objs.size(); ++i)
            overlay(objs[i].bound).setTo(_colors[colors[i]], objs[i].mask);

        cv::Mat out;
        cv::addWeighted(image, 0.45, overlay, 0.55, 0, out);

        for (size_t i = 0; i < objs.size(); ++i)
        {
            const auto &obj = objs[i];
            const auto &color = _colors[colors[i]];
            cv::rectangle(out, obj.bound, color, 1);

            std::string label = cv::format("%s %.2f", _classNames[obj.id].c_str(), obj.accu);
            int baseline = 0;
            cv::Size size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
            cv::Point org(obj.bound.x, std::max(obj.bound.y, size.height + 4));
            cv::rectangle(out, cv::Rect(org.x, org.y - size.height - 4, size.width + 4, size.height + 4), color, cv::FILLED);
            cv::putText(out, label, org + cv::Point(2, -3), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }
        return out;
    }
}
