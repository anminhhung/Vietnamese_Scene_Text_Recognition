pip3 install -r requirements.txt
# EfficientNet mask-rcnn
gdown --id 1S5SCobkYG5geuJZ7H_SzpaZ6FY8ZgRpY
# VietOCR
# gdown --id 19QwrIEe4FbO3oaxR-21nY5BkbkKRfLO8
gdown --id 112dP9jSh6VMw_zpGidjCoXld_2kBlhlp
# resnext
gdown --id 1Olv4QZMjkhCRKSUb3lyzFTdUyUwABgTm
# MaskRCNN backbone python file
gdown --id 1oqJ3qCRvWlHBUHMMy27JQxUVGa-6tWGU
# TestA
gdown --id 1sUqBG2mTNIovZ_zBUj37A1TtCTzSUEzs
unzip TestA.zip 

# move model 
mv efficientb7_fail.pth models/
mv resnext50_fail.pth models/
mv resnet101_crop.pth models/
mv transformerocr.pth models/vietocr
mv TestA data