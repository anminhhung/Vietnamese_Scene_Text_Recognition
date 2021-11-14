pip3 install -r requirements.txt
# EfficientNet mask-rcnn
gdown --id 1FQz_CHOmv5WOTJpH8TIZEV-MKy38qH2p
# VietOCR
gdown --id 19QwrIEe4FbO3oaxR-21nY5BkbkKRfLO8
# resnext
gdown --id 1Olv4QZMjkhCRKSUb3lyzFTdUyUwABgTm
# TestA
gdown --id 1sUqBG2mTNIovZ_zBUj37A1TtCTzSUEzs
unzip TestA.zip 

# move model 
mv efficientb7_fail.pth models/
mv resnext50_fail.pth models/
mv transformerocr.pth models/vietocr
mv TestA data