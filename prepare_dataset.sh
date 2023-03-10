
mkdir ./data_root0
# download all zip files and unzip into data_root0
cd data_root0
unzip TRAIN_RAD_H.zip
unzip TRAIN_CAM_0.zip
unzip TEST_RAD_H.zip
unzip TRAIN_RAD_H_ANNO.zip
unzip CAM_CALIB.zip

# make folders for data and annotations
mkdir sequences
mkdir annotations

# rename unzipped folders
mv TRAIN_RAD_H sequences/train
mv TRAIN_CAM_0 train
mv TEST_RAD_H sequences/test
mv TRAIN_RAD_H_ANNO annotations/train

# merge folders and remove redundant
rsync -av train/ sequences/train/
rm -r train


python ./RadarFormer/tools/prepare_dataset/prepare_data.py \
--config ./RadarFormer/configs/config_rodnet_hg1v2_win16_mnet.py \
--data_root ./data_root0/ --split train,test --out_data_dir ./Pickle0

