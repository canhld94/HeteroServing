from client import http_client
from img_split_join import splitbase
import numpy as np
from timeit import default_timer as timer


class App():
  def __init__(self,srcpath,ip,port):
    self.splitbase = splitbase(srcpath=srcpath)
    self.client = http_client(ip=ip,port=port)

  def run(self, name, ext):
    start = timer()
    patch_list = self.splitbase.SplitSingle(name=name,rate=1,extent=ext)
    print(len(patch_list))
    print(timer()-start)
    self.client.send_inference_mp(patch_list=patch_list)
    print(timer()-start)

if __name__ == '__main__':
    # srcpath: path to image folder
    # ip & port
    app = App(srcpath = r'imgs', ip=r'localhost', port='8081')
    # name: name of the image we want to run
    # ext: image extension
    app.run(name=r'oils_high_res',ext='.jpg')