# How to configure the inference server

Do **NOT** add comment to the configure file

```JSON
{
  "ip": "0.0.0.0",            // ip of the server
  "port": "8081",             // port of the server
  "inference engines": [
    {
      "device": "intel cpu",  // Device, currently support 'intel cpu, intel fpga, nvidia gpu'
      "replicas": "1",        // Number of incerence engine you want to create on this device
      "model": {}             // Parametter to create models, it's all depend you to include any
                              // parametter that help you to create the engine. Note that the
                              // input of factory method to create inference engine is a Boost
                              // property tree object, which is this node
    },
    {
      "device": "intel fpga",
      "replicas": "1",
      "model": {},
      "bitstream": "/opt/intel/openvino_2019.1.144/bitstreams/a10_devkit_bitstreams/2019R1_A10DK_FP16_MobileNet_Clamp.aocx"
    },
    {
      "device": "nvidia gpu",
      "replicas": "1",
      "model": {
        "name": "ssd",
        "graph": "/home/canhld/workplace/InferenceServer/onnx_model/model_i32.onnx",
        "input nodes": "",
        "input shape": "",
        "output nodes": [],
        "label": "/home/canhld/workplace/InferenceServer/openvino_models/dota/dota_ssd.txt"
      }
    }
  ]
}
```
