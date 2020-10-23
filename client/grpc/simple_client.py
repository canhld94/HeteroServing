import grpc
import inference_rpc_pb2_grpc
import inference_rpc_pb2
import sys
import json
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from timeit import default_timer as timer


def simple_client(image, ip, port):
  # header must specify content-type is image/xxx
  # can be any type of images, but it should perfectly be jpeg file with size < 1MB
  # I'm setting max reading file from socket is 1MB, there are a bug make reading > 1M 
  # file take long time

  headers = {'Content-Type': 'image/jpeg'}

  # No need to decode, just read the raw byte and send
  f = open(image,'rb')
  body = f.read()
  conn = grpc.insecure_channel(ip+":"+str(port))
  stub = inference_rpc_pb2_grpc.inference_rpcStub(conn)
  res = stub.run_detection(inference_rpc_pb2.encoded_image(data=body,size=len(body)))

  # Parse response
  im = Image.open(f)
  draw = ImageDraw.Draw(im)
  for p in res.bboxes:
      label_id = p.label_id
      label = p.label
      score = p.prob
      bbox = p.box
      tl = (int(bbox.xmin),int(bbox.ymin))
      br = (int(bbox.xmax),int(bbox.ymax))
      draw.rectangle((tl,br),outline = "red", width = 3)
      draw.text((int(bbox.xmin),int(bbox.ymin)-10), label + " " + str(round(score,2)))

  im.save("testing.jpg","jpeg")

if __name__ == '__main__':
  start = timer()
  simple_client(image=r'../imgs/AirbusDrone.jpg', ip=r'localhost', port=8081)
  print(timer() - start)