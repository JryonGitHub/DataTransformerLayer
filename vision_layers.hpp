#ifndef CAFFE_VISION_LAYERS_HPP_
#define CAFFE_VISION_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/confusion_matrix.hpp"

namespace caffe {



template <typename Dtype>
class DataTransformerLayer : public Layer<Dtype> {
 public:
  explicit DataTransformerLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "DataTranformer"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void FillBlob_Gassian_Filter(Blob<Dtype>& gaussian_kernel_mtx, Dtype radius);
  void Initialize_Elastic_Mtx(Blob<Dtype>& elastic_distortion_mtx_0, Blob<Dtype>& gaussian_kernel_mtx,Blob<Dtype>& elastic_distortion_mtx_1, Dtype amplitude, Dtype radius);
  Blob<Dtype> _colordistortion_delta1,_colordistortion_delta2,
      _colordistortion_delta3,_colordistortion_delta4;
  Dtype _colordistortion_delta1_sigma,_colordistortion_delta2_sigma,
      _colordistortion_delta3_sigma,_colordistortion_delta4_sigma;
  Dtype _rotate_angle_scope,_translation_w_scope,_translation_h_scope,
      _scale_w_scope,_scale_h_scope;
  Blob<Dtype> _rotate_angle_info,_translation_w_info,_translation_h_info,
      _scale_w_info,_scale_h_info;
  bool _h_flip;
  Blob<Dtype> _h_flip_indicator;

  Dtype _amplitude;// = this->layer_param_.data_transformer_l_param().amplitude();
  Dtype _radius;
  bool _elastic_transform;
  Blob<Dtype> _elastic_distortion_mtx_0,_gaussian_kernel_mtx,_elastic_distortion_mtx_1;
  Blob<Dtype> _temp_mtx_0,_temp_mtx_1;

  int _canvas_h,_canvas_w;
  Blob<Dtype> _new_canvas;
};


}  // namespace caffe

#endif  // CAFFE_VISION_LAYERS_HPP_
