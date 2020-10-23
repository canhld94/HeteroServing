import os
import cv2
import sys
import json
import mimetypes
import http.client
import numpy as np
from multiprocessing import Pool
from functools import partial
from timeit import default_timer as timer

class splitbase():
    def __init__(self,
                 srcpath,
                 gap=0,
                 subsize=1024,
                 padding=True):
        self.srcpath = srcpath
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.padding = padding

    def SplitSingle(self, name, rate, extent):
        img = cv2.imread(os.path.join(self.srcpath, name + extent))
        assert np.shape(img) != ()

        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        else:
            resizeimg = img

        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]
        print(np.shape(resizeimg))

        patch_list = []
        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                subimg = img[up: (up + self.subsize), left: (left + self.subsize)]
                h, w, c = np.shape(subimg)
                if (self.padding):
                    outimg = np.zeros((self.subsize, self.subsize, 3))
                    outimg[0:h, 0:w, :] = subimg
                patch_list.append(outimg)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide
        return patch_list



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
    _, body = cv2.imencode('.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY), 50])
    conn = http.client.HTTPConnection(self.ip+":"+self.port)
    # POST to /inference with body is the imagesss
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


class App():
  def __init__(self,srcpath,ip,port):
    self.splitbase = splitbase(srcpath=srcpath)
    self.client = http_client(ip=ip,port=port)

  def run(self, name, ext):
    start = timer()
    patch_list = self.splitbase.SplitSingle(name=name,rate=1,extent=ext)
    print(len(patch_list))
    print(timer()-start)
    # start = timer()
    self.client.send_inference_mp(patch_list=patch_list)
    print(timer()-start)

if __name__ == '__main__':
    # srcpath: path to image folder
    # ip & port
    app = App(srcpath = r'../imgs', ip=r'localhost', port=r'8080')
    # name: name of the image we want to run
    # ext: image extension
    app.run(name=r'AirbusDrone',ext=r'.jpg')