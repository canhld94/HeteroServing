import http.client
import sys
import json
import cv2
from multiprocessing import Pool
from functools import partial
import mimetypes



def send_inference_wrap(img, http_client):
    http_client.send_inference(img)

class http_client():
  def __init__(self, ip, port, num_process = 16):
    self.ip = ip
    self.port = port
    self.pool = Pool(num_process)
  
  def send_inference(self, img):
    # header must specify content-type is image/xxx
    # can be any type of images, but it should perfectly be jpeg file with size < 1MB
    # I'm setting max reading file from socket is 1MB, there are a bug make reading > 1M 
    # file take long time
    headers = {'Content-Type': 'image/jpg'}
    # encode img 
    _, body = cv2.imencode('.jpg',img)
    conn = http.client.HTTPConnection(self.ip+":"+self.port)
    # POST to /inference with body is the image
    conn.request('POST','/inference', body.tobytes(), headers)
    res = conn.getresponse()
    jsondata = res.read()
    # print(jsondata)
    # data = json.loads(jsondata.decode('utf-8'))
    # print(data)
    # TODO: read and parse (and merge) the result

  def send_inference_mp(self,patch_list):
    worker = partial(send_inference_wrap, http_client=self)
    self.pool.map(worker,patch_list)

  def __getstate__(self):
      self_dict = self.__dict__.copy()
      del self_dict['pool']
      return self_dict

  def __setstate__(self, state):
      self.__dict__.update(state)

# # Parse response
# data = json.loads(jsondata.decode('utf-8'))
# im = Image.open(body)
# draw = ImageDraw.Draw(im)
# for p in data["predictions"]:
#     label_id = p["label_id"]
#     label = p["label"]
#     score = float(p["confidences"])
#     bbox = p["detection_box"]
#     tl = (int(bbox[0]),int(bbox[1]))
#     br = (int(bbox[2]),int(bbox[3]))
#     draw.rectangle((tl,br),outline = "red", width = 3)
#     draw.text((int(bbox[0]),int(bbox[1])-10), label + " " + str(round(score,2)))

# im.save("testing.jpg","jpeg")

"""
Response format:

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

"""