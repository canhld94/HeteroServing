# How to configure the inference server

Do **NOT** add comment to the configure file

```JSON
{
  "ip": "0.0.0.0",            // ip of the server
  "port": "8081",             // port of the server
  "protocol": "grpc",         // protocol, http or grpc
  "inference engines": [
    {
      "device": "intel cpu",  // Device, currently support 'intel cpu, intel fpga, nvidia gpu'
      "replicas": "1",        // Number of inference engine you want to create on this device
      "model": {
        // Tree mandatory fields are: 'name', 'graph', and 'label'.
        // In addition, it's all depend you to include any
        // parameter that help you to create the engine. Note that the
        // input of factory method to create inference engine is a Boost
        // property tree object, which is this node
        "name": "ssd",
        "graph": "deploy/openvino_model/DOTA/CPU/ssd_mobilenet_v2.xml",
        "label": "deploy/label/dota_v2.txt"

      }
    },
    {
      "device": "intel fpga",
      "replicas": "1",
      "model": {
        "name": "ssd",
        "graph": "",
        "label": ""
      },
      "bitstream": ""
    },
    {
      "device": "nvidia gpu",
      "replicas": "1",
      "model": {
        "name": "ssd",
        "graph": "",
        "label": ""
      }
    }
  ]
}
```
