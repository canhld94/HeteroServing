# FPGA Inference Server

The FPGA Inference Server run SSD on FPGA (and CPU) and expose RESTFul APIs to end-users to utilize the infrence engine in client-server model. This is very similar to [Tensorflow Serving](https://github.com/tensorflow/serving), however, TF serving doesn't support FPGA back-end.

**IMPORTANT**: Developing server at `143.248.148.118:8080`

## Request

The endpoint of inference engine is `POST /inference`

Inference request format: the inference request upload an image in any format and recieve the detection result in the `JSON` format.

Example with `curl` command line: (tested with curl 7.47.0 @ ubuntu 16.04)

```bash
curl "http://143.248.148.118:8080/inference" \
        -X POST \
        --data-binary "@img0.jpeg" # replace with your file
        -H "Content-Type: image/jpeg"
```

Currently, the size of the image must < 1MB due to a bug in reading from socket.

## Response

- `label_id`: label_id
- `label`: label name
- `confidences`: confidence of the detection
- `detection_box`: detection box `x_min,y_min,x_max,y_max` with `top_left` of the image is `(0,0)`

```json
{
  "status": "ok",
  "predictions": [
    {
      "label_id": "3",
      "label": "car",
      "confidences": "0.905718684",
      "detection_box": ["1176", "723", "1232", "750"]
    },
    {
      "label_id": "3",
      "label": "car",
      "confidences": "0.787045956",
      "detection_box": ["301", "725", "346", "752"]
    },
    {
      "label_id": "3",
      "label": "car",
      "confidences": "0.647054553",
      "detection_box": ["1169", "449", "1228", "476"]
    }
  ]
}
```
