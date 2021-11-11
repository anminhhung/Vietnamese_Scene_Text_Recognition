pip3 install -r requirements.txt
# Resnet mask-rcnn
gdown --id 1-1YaQktobQPTauj74CZLRMTKmOnT2jqa
# VietOCR
gdown --id 19QwrIEe4FbO3oaxR-21nY5BkbkKRfLO8
# TestA
gdown --id 1sUqBG2mTNIovZ_zBUj37A1TtCTzSUEzs
unzip TestA.zip 

# move model 
mv resnext50.pth models/mask_rcnn
mv transformerocr.pth models/vietocr
mv TestA data