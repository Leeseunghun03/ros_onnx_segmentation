#include <ros_onnx_segmentation/yolo_processor.h>

#include <algorithm>
#include <cmath>
#include <map>
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

        cv::RNG rng(12345);
        for (int i = 0; i < num_classes; ++i)
            _colors.emplace_back(rng.uniform(64, 256), rng.uniform(64, 256), rng.uniform(64, 256));

        _objectPub = _node->create_publisher<seg_msgs::msg::ObjectInfoArray>("object_info_array", 10);
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

        seg_msgs::msg::ObjectInfoArray array_msg;
        array_msg.header = header;
        std::map<int, size_t> groups;
        std::vector<Obj> objs;

        for (int idx : keep)
        {
            Obj obj;
            obj.id = class_ids[idx];
            obj.accu = scores[idx];
            obj.bound = boxes[idx];

            cv::Mat coeffs(1, _maskChannels, CV_32F, det.ptr<float>(rows[idx]) + 4 + num_classes);
            obj.mask = get_mask(coeffs, protos, info, obj.bound);

            auto it = groups.find(obj.id);
            if (it == groups.end())
            {
                seg_msgs::msg::ObjectInfo info_msg;
                info_msg.class_name = _classNames[obj.id];
                array_msg.objects.push_back(info_msg);
                it = groups.emplace(obj.id, array_msg.objects.size() - 1).first;
            }
            auto &group = array_msg.objects[it->second];
            group.object_count += 1;
            group.x_position.push_back(obj.bound.x);
            group.y_position.push_back(obj.bound.y);
            group.width.push_back(obj.bound.width);
            group.height.push_back(obj.bound.height);

            objs.push_back(std::move(obj));
        }

        _objectPub->publish(array_msg);

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

    cv::Mat YoloProcessor::draw_result(const cv::Mat &image, const std::vector<Obj> &objs) const
    {
        cv::Mat overlay = image.clone();
        for (const auto &obj : objs)
            overlay(obj.bound).setTo(_colors[obj.id], obj.mask);

        cv::Mat out;
        cv::addWeighted(image, 0.6, overlay, 0.4, 0, out);

        for (const auto &obj : objs)
        {
            cv::rectangle(out, obj.bound, _colors[obj.id], 2);
            std::string label = cv::format("%s %.2f", _classNames[obj.id].c_str(), obj.accu);
            cv::Point org(obj.bound.x, std::max(obj.bound.y - 4, 12));
            cv::putText(out, label, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, _colors[obj.id], 1);
        }
        return out;
    }
}
