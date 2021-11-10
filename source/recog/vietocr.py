import cv2
import time 

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from shapely.geometry import Polygon

from libs import CRAFT_DETECTOR
from utils.create_output_file import create_output
from configs.config import init_config

# config
CFG = init_config()
DEVICE = CFG["device"]["device"]
craft_weight_path = CFG["craft"]["weight_path"]
vietocr_weight_path = CFG["vietocr"]["weight_path"]

# TEXT_DETECTOR
TEXT_DETECTOR = CRAFT_DETECTOR(craft_weight_path, DEVICE)

# TEXT RECOGNITION
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = vietocr_weight_path
config['cnn']['pretrained']=False
# config['device'] = 'cuda:0' 
config['device'] = 'cpu' 
config['predictor']['beamsearch']=False

TEXT_RECOGNIZER= Predictor(config)

def predict_image(image_path, result_dir="dataset/results"):
    image_name = (image_path.split("/")[-1]).split(".")[0]
    image = cv2.imread(image_path)
    image_visual = image.copy()

    time_start =  time.time()
    # detect 
    horizontal_list, free_list = TEXT_DETECTOR.detect(image)
    list_contents = []
    list_text = []
    for bbox in horizontal_list:
        crop_image = TEXT_DETECTOR.crop_image(image, bbox)

        result_text = TEXT_RECOGNIZER.predict(crop_image)
        
        content = "{},{},{},{},{},{},{},{},{}\n".format(
                bbox[0], bbox[2], bbox[1], bbox[2], bbox[1], bbox[3], bbox[0], bbox[3], result_text)
        list_contents.append(content)
        
        list_text.append(result_text)
    
    with open("result.txt", "a+") as f:
        for text in list_text:
            f.write("{}\n".format(text))

    # write text
    # create_output(image_name, list_contents, result_dir)

    # print("Time process: ", time.time() - time_start)
