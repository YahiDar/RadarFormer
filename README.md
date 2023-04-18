# RadarFormer: Lightweight and Accurate Real-Time Radar Object Detection Model

This is the official implementation of the RadarFormer paper, based on the [RODNet](https://github.com/yizhou-wang/RODNet#rodnet-radar-object-detection-network) implementation.
Please do cite our work if this repository helps your research.

```
@misc{dalbah2023radarformer,
      title={RadarFormer: Lightweight and Accurate Real-Time Radar Object Detection Model}, 
      author={Yahia Dalbah and Jean Lahoud and Hisham Cholakkal},
      year={2023},
      eprint={2304.08447},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

This code is heavily reliant on the RODNet repository, please do visit and cite their works as well if you are interested in this work.
```
@inproceedings{wang2021rodnet,
  author={Wang, Yizhou and Jiang, Zhongyu and Gao, Xiangyu and Hwang, Jenq-Neng and Xing, Guanbin and Liu, Hui},
  title={RODNet: Radar Object Detection Using Cross-Modal Supervision},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  month={January},
  year={2021},
  pages={504-513}
}
```
```
@article{wang2021rodnet,
  title={RODNet: A Real-Time Radar Object Detection Network Cross-Supervised by Camera-Radar Fused Object 3D Localization},
  author={Wang, Yizhou and Jiang, Zhongyu and Li, Yudong and Hwang, Jenq-Neng and Xing, Guanbin and Liu, Hui},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  volume={15},
  number={4},
  pages={954--967},
  year={2021},
  publisher={IEEE}
}
```


## Installation

Clone RadarFormer code.
```commandline
cd $RadarFormer
git clone https://github.com/YahiDar/RadarFormer.git
```

Create a conda environment for RadarFormer using the provided env.yml file.
```commandline
conda env create -n RadarFormer --file env.yml
conda activate RadarFormer
```

After that, clone the ```cruw-devkit``` repository into the same directory and install it using

```commandline
git clone https://github.com/yizhou-wang/cruw-devkit
cd cruw-devkit
pip install .
cd ..
```

Then setup the RODNet package by:
```commandline
pip install -e .
```

To run the MaXViT based model, you need to install the code from the [MaxViT Repository](https://github.com/ChristophReich1996/MaxViT) through:

```commandline
pip install git+https://github.com/ChristophReich1996/MaxViT
```

NOTE: This environment does NOT include the bare minimum required libraries, and includes libraries used in further research that will be published soon. A link to that research will be provided.



## Prepare the dataset

Edit and run the ```prepare_dataset.sh``` file with desired directories.

## Train models

Either edit and run the ```train.sh``` file with desired directories, or use:

```commandline
python tools/train.py --config configs/<CONFIG_FILE> \
        --data_dir data/<DATA_FOLDER_NAME> \
        --log_dir checkpoints/
```

You might need to change line 152 in the ```.\rodnet\datasets\loaders\CRDataset.py``` file based on data directory.

## Testing models

Either edit and run the ```test.sh``` file with desired directories, or use:

```commandline
python tools/test.py --config configs/<CONFIG_FILE> \
        --data_dir data/<DATA_FOLDER_NAME> \
        --checkpoint <CHECKPOINT_PATH> \
        --res_dir results/
```

You might need to change the base_root/data_root/anno_root paths in the config files.

IMPORTANT NOTE:

The test split annotations are NOT provided, and you have to use the online [RODNet challenge](https://codalab.lisn.upsaclay.fr/competitions/1063#participate-submit_results) website to test it.

To do so, you have to export the annotations (output of the test.py file) using the ```./tools/format_transform/convert_rodnet_to_rod2021.py``` file. The output must be zipped and uploaded to the challenge website. 

The ```test.sh``` file automates the process.

The pretrained MaXViT model weights can be downloaded from:

https://mbzuaiac-my.sharepoint.com/:u:/g/personal/yahia_dalbah_mbzuai_ac_ae/EZ5RVt7nrK5OgozBs200hDQBIZqsGdf2bkJrwEE2jQ4KOw?e=cjf0qo
