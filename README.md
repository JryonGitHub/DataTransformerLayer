# DataTransformerLayer

CAFFE IMPLEMENTATION OF data augumentation (CUDA version)
Deduction and results are in folder [./attachments]
proto definition:
```
message DataTranformerLParameter {
  //output-canvas size, default is correspond to input size
  optional int32 canvas_w = 14 [default = -1];
  optional int32 canvas_h = 15 [default = -1];
  //absolute distortion
  optional float delta1_sigma = 1 [default = 12.75];
  //relative distortion
  optional float delta2_sigma = 2 [default = 0.15];
  //height-wise distortion
  optional float delta3_sigma = 3 [default = 0];
  //width-wise distortion
  optional float delta4_sigma = 4 [default = 0];
 
  optional float rotate_angle_scope = 5 [default = 0.349];//20 degree
  optional float translation_w_scope = 6 [default = 8];
  optional float translation_h_scope = 7 [default = 8];
  optional float scale_w_scope = 8 [default = 1.2];
  optional float scale_h_scope = 9 [default = 1.2];
   
  // do mirror
  optional bool h_flip = 10 [default = true];
  // elastic transform
  optional bool elastic_transform = 13 [default = false];
  optional float amplitude = 11 [default = 1];
  optional float radius = 12 [default = 1];
}
```

Usage:
```
name: "CIFAR10_full"
layer {
  name: "cifar"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mean_file: "../../data/mean.binaryproto"
  }
  data_param {
    source: "../../data/cifar10_train_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
layer {
  name: "cifar"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mean_file: "../../data/mean.binaryproto"
  }
  data_param {
    source: "../../data/cifar10_test_lmdb"
    batch_size: 10
    backend: LMDB
  }
}
layer {
    name: "data_transformer"
    type: "DataTransformer"
    bottom: "data"
    top: "data"
    include {
        phase: TRAIN
    }
    data_transformer_l_param {
        delta1_sigma: 10.5
        delta2_sigma: 0.15
        delta3_sigma: 1.1 #2.4
        delta4_sigma: 1.1 #2.4
        rotate_angle_scope: 0.349 #20degree
        translation_w_scope: 16
        translation_h_scope: 16
        scale_w_scope: 1.2
        scale_h_scope: 1.2
        h_flip: true
        elastic_transform: false
        amplitude: 1
        radius: 1
    }
}
//.......
```
