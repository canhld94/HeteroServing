import http.client
import sys
import json
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

def simple_client(image, ip, port):
  # header must specify content-type is image/xxx
  # can be any type of images, but it should perfectly be jpeg file with size < 1MB
  # I'm setting max reading file from socket is 1MB, there are a bug make reading > 1M 
  # file take long time

  headers = {'Content-Type': 'image/jpeg'}

  # No need to decode, just read the raw byte and send
  body = open(sys.argv[1],'rb')
  ip = sys.argv[2]
  port = sys.argv[3]
  conn = http.client.HTTPConnection(ip+":"+str(port))

  # POST to /inference with body is the image
  conn.request('POST','/inference', body, headers)
  res = conn.getresponse()
  jsondata = res.read()

  # Parse response
  data = json.loads(jsondata.decode('utf-8'))
  im = Image.open(body)
  draw = ImageDraw.Draw(im)
  for p in data["predictions"]:
      label_id = p["label_id"]
      label = p["label"]
      score = float(p["confidences"])
      bbox = p["detection_box"]
      tl = (int(bbox[0]),int(bbox[1]))
      br = (int(bbox[2]),int(bbox[3]))
      draw.rectangle((tl,br),outline = "red", width = 3)
      draw.text((int(bbox[0]),int(bbox[1])-10), label + " " + str(round(score,2)))

  im.save("testing.jpg","jpg")

if __name__ == '__main__':
  simple_client(image=r'imgs/AirbusDrone.jpg', ip=r'localhost', port=8081)