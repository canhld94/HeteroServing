# FPGA Inference Server

The FPGA Inference Server run SSD on FPGA (and CPU) and expose RESTFul APIs to end-users to utilize the infrence engine in client-server model. This is very similar to [Tensorflow Serving](https://github.com/tensorflow/serving), however, TF serving doesn't support FPGA back-end.

**IMPORTANT**: 

Developing server at `143.248.148.118:8080`

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
curl "http://143.248.148.118:8080/inference" \
        -X POST \
        --data-binary "@img0.jpeg" # replace with your file
        -H "Content-Type: image/jpeg"
```

`HTTP`: (tested with advance rest client on Window 10)

```http
POST /inference HTTP/1.1
Host: 143.248.148.118:8080
Content-Type: image/jpeg

[object File]
```

`Javascript`: (tested with nodejs 12.18.3)

```javascript
const http = require("http");
const fs = require("fs");
const init = {
  host: "143.248.148.118",
  path: "/inference",
  port: 8080,
  method: "POST",
  headers: {
    "Content-Type": "image/jpeg",
  },
};
const callback = function (response) {
  let result = Buffer.alloc(0);
  response.on("data", function (chunk) {
    result = Buffer.concat([result, chunk]);
  });

  response.on("end", function () {
    // result has response body buffer
    console.log(result.toString());
  });
};

const req = http.request(init, callback);
const body = fs.readFileSync("img0.jpeg"); // replace with your file
req.write(body);
req.end();
```

`Python`: (tested with python 3.7)

```python
import http.client

headers = {'Content-Type': 'image/jpeg'}
body = open('img0.jpeg','rb') #replace with your file

conn = http.client.HTTPConnection('143.248.148.118:8080')
conn.request('POST','/inference', body, headers)
res = conn.getresponse()

data = res.read()
print(res.status, res.reason)
print(data.decode('utf-8'))
print(res.getheaders())
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


Azure response:

```JSON
{
   "objects":[
      {
         "rectangle":{
            "x":730,
            "y":66,
            "w":135,
            "h":85
         },
         "object":"kitchen appliance",
         "confidence":0.501
      },
      {
         "rectangle":{
            "x":523,
            "y":377,
            "w":185,
            "h":46
         },
         "object":"computer keyboard",
         "confidence":0.51
      },
      {
         "rectangle":{
            "x":471,
            "y":218,
            "w":289,
            "h":226
         },
         "object":"Laptop",
         "confidence":0.85,
         "parent":{
            "object":"computer",
            "confidence":0.851
         }
      },
      {
         "rectangle":{
            "x":654,
            "y":0,
            "w":584,
            "h":473
         },
         "object":"person",
         "confidence":0.855
      }
   ],
   "requestId":"a7fde8fd-cc18-4f5f-99d3-897dcd07b308",
   "metadata":{
      "width":1260,
      "height":473,
      "format":"Jpeg"
   }
}
```

GCP response example:

```JSON
{
  "responses": [
    {
      "localizedObjectAnnotations": [
        {
          "mid": "/m/01bqk0",
          "name": "Bicycle wheel",
          "score": 0.89648587,
          "boundingPoly": {
            "normalizedVertices": [
              {
                "x": 0.32076266,
                "y": 0.78941387
              },
              {
                "x": 0.43812272,
                "y": 0.78941387
              },
              {
                "x": 0.43812272,
                "y": 0.97331065
              },
              {
                "x": 0.32076266,
                "y": 0.97331065
              }
            ]
          }
        },
        {
          "mid": "/m/0199g",
          "name": "Bicycle",
          "score": 0.886761,
          "boundingPoly": {
            "normalizedVertices": [
              {
                "x": 0.312,
                "y": 0.6616471
              },
              {
                "x": 0.638353,
                "y": 0.6616471
              },
              {
                "x": 0.638353,
                "y": 0.9705882
              },
              {
                "x": 0.312,
                "y": 0.9705882
              }
            ]
          }
        },
        {
          "mid": "/m/01bqk0",
          "name": "Bicycle wheel",
          "score": 0.6345275,
          "boundingPoly": {
            "normalizedVertices": [
              {
                "x": 0.5125398,
                "y": 0.760708
              },
              {
                "x": 0.6256646,
                "y": 0.760708
              },
              {
                "x": 0.6256646,
                "y": 0.94601655
              },
              {
                "x": 0.5125398,
                "y": 0.94601655
              }
            ]
          }
        },
        {
          "mid": "/m/06z37_",
          "name": "Picture frame",
          "score": 0.6207608,
          "boundingPoly": {
            "normalizedVertices": [
              {
                "x": 0.79177403,
                "y": 0.16160682
              },
              {
                "x": 0.97047985,
                "y": 0.16160682
              },
              {
                "x": 0.97047985,
                "y": 0.31348917
              },
              {
                "x": 0.79177403,
                "y": 0.31348917
              }
            ]
          }
        },
        {
          "mid": "/m/0h9mv",
          "name": "Tire",
          "score": 0.55886006,
          "boundingPoly": {
            "normalizedVertices": [
              {
                "x": 0.32076266,
                "y": 0.78941387
              },
              {
                "x": 0.43812272,
                "y": 0.78941387
              },
              {
                "x": 0.43812272,
                "y": 0.97331065
              },
              {
                "x": 0.32076266,
                "y": 0.97331065
              }
            ]
          }
        },
        {
          "mid": "/m/02dgv",
          "name": "Door",
          "score": 0.5160098,
          "boundingPoly": {
            "normalizedVertices": [
              {
                "x": 0.77569866,
                "y": 0.37104446
              },
              {
                "x": 0.9412425,
                "y": 0.37104446
              },
              {
                "x": 0.9412425,
                "y": 0.81507325
              },
              {
                "x": 0.77569866,
                "y": 0.81507325
              }
            ]
          }
        }
      ]
    }
  ]
}
```