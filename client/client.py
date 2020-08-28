import http.client
import sys
import json
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

headers = {'Content-Type': 'image/jpeg'}
body = open(sys.argv[1],'rb')

conn = http.client.HTTPConnection(sys.argv[2])
conn.request('POST','/inference', body, headers)
res = conn.getresponse()

jsondata = res.read()

data = json.loads(jsondata.decode('utf-8'))
im = Image.open(body)
draw = ImageDraw.Draw(im)
for p in data["predictions"]:
    label = p["label"]
    score = p["confidences"]
    bbox = p["detection_box"]
    tl = (int(bbox[0]),int(bbox[1]))
    br = (int(bbox[2]),int(bbox[3]))
    draw.rectangle((tl,br),outline = "red", width = 5)

im.save("testing.jpg","JPEG")