
/*
    Generic inference engine interface
*/
class inference_engine {

};

/*
    Generic object detection interface
*/

class object_detection {

};

/*
    SSD inferencer
*/

class ssd : public object_detection {

};

/*
    Yolo inferencer
*/

class yolo : public object_detection {

};

/*
    Fast r cnn inferencer
*/

class fast_r_cnn : public object_detection {

};

/*
    classification generic interface
*/

class classification : public inference_engine {

};

/*
    Resnet inferencer
*/

class resnet101 : public classification {

};

/*
    Segmentation generic interface
*/

class segmentation : public inference_engine {
    
};

/*
    MaskRCNN inferencer
*/

class mask_r_cnn :  public segmentation {

};