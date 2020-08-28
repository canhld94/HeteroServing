# How to configure the inference server

Do not add comment to the configure file

```JSON
{
  "ip": "IP address",
  "port": "Port you want to run",
  "inference engine": [
    {
      "name": "Name of Inference Engine",
      "device": "Device to run, either CPU, GPU, or HETERO:FPGA,CPU",
      "model": "path to the xml model",
      "labels": "path to the label file",
      "fpga configuration": {
        "dev": "device number",
        "bitstream": "path to the bitstream will be configured on FPGA"
      }
    }
  ]
}
```
