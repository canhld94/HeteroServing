# How to configure the inference server

Do not add comment to the configure file

```JSON
{
  "ip": "your ip address",
  "port": "your port",
  "inference engines": [
    {
      "device": "cpu",
      "models": [
        {
            "name": "name of the models",
            "graph": "computation graph",
            "label": "label file",
            "replicas": "number of inference engines you want to create"
        }
      ]
    },
    {
      "device": "fpga",
      "models": [
          {
              "name": "name of the models",
              "graph": "computation graph",
              "label": "label file",
              "replicas": "number of inference engines you want to create"
          }
      ],
      "bitstream": "associated bitstream"
    },
    {
      "device": "gpu",
      "models": [
          {
              "name": "name of the models",
              "graph": "computation graph",
              "label": "label file",
              "replicas": "number of inference engines you want to create"
          }
      ]
    }
  ]
}
```
