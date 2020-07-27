# FPGA Inference Server

The FPGA Inference Server run SSD on FPGA (and CPU) and expose RESTFul APIs to end-users to utilize the infrence engine in client-server model. This is very similar to [Tensorflow Serving](https://github.com/tensorflow/serving), however, TF serving doesn't support FPGA back-end.

__IMPORTANT__: Developing server at `143.248.148.118:8081`

## API specificiation

Currently, the server only support three APIs:

- `GET /` --> greating messagge
- `GET /metadata` --> get the metadata of the model | _under developement_
- `POST /inference` --> the endpoint of inference engine | _beta release, under testing_

The detail format of request and response is as follow.

### Request format

The two `/GET` request are trivial and we can temporary ignore it in the development phase.

Inference request format: the inference request upload an image in any format and recieve the detection result in the `JSON` format. Note that the `content-type` field in the header must match the image format (or encoding method).

`curl` command line: (tested with curl 7.47.0 @ ubuntu 16.04)

```bash
curl "http://143.248.148.118:8081/inference" \
        -X POST \
        --data-binary "@img0.jpeg" # replace with your file
        -H "Content-Type: image/jpeg"
```

`HTTP`: (tested with advance rest client on Window 10)

```http
POST /inference HTTP/1.1
Host: 143.248.148.118:8081
Content-Type: image/jpeg

[object File]
```

`Javascript`: (tested with nodejs 12.18.3)

```javascript
const http = require('http');
const fs = require('fs')
const init = {
  host: '143.248.148.118',
  path: '/inference',
  port: 8081,
  method: 'POST',
  headers: {
    'Content-Type': 'image/jpeg',
  }
};
const callback = function(response) {
  let result = Buffer.alloc(0);
  response.on('data', function(chunk) {
    result = Buffer.concat([result, chunk]);
  });

  response.on('end', function() {
    // result has response body buffer
    console.log(result.toString());
  });
};

const req = http.request(init, callback);
const body = fs.readFileSync('img0.jpeg'); // replace with your file
req.write(body);
req.end();
```

`Python`: (tested with python 3.7)

```python
import http.client

headers = {'Content-Type': 'image/jpeg'}
body = open('img0.jpeg','rb') #replace with your file

conn = http.client.HTTPConnection('143.248.148.118:8081')
conn.request('POST','/inference', body, headers)
res = conn.getresponse()

data = res.read()
print(res.status, res.reason)
print(data.decode('utf-8'))
print(res.getheaders())
~

```

### Response format

Example response of `GET /`

```json
{
    "type": "greeting",
    "from": "canhld@kaist.ac.kr",
    "message": "welcome to SSD inference server version 1",
    "what next": "GET /v1/ for supported API"
}
```

Example response of `GET /metadata`

```json
{
    "from": "canhld@kaist.ac.kr",
    "message": "this is metadata request"
}
```

Example response of `POST /inference`

- `label_id`: label_id in MSCOCO dataset
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
