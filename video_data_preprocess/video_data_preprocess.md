# Data Preparation

## Nuplan
We use [NuPlan](https://nuplan.org/) for training and testing. We organize our training and testing datasets as follows.

### Download
Please download all the splits in [NuPlan](https://nuplan.org/). We follow [NuPlan-Download-CLI](https://github.com/Syzygianinfern0/NuPlan-Download-CLI) to download all the splits. Once you download all the files, please `unzip` them first.

### Reorganize
Please move your files and make sure that they are organized like this:
```
nuplan-v1.1
├── splits
│     ├── mini
│     │    ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
│     │    ├── 2021.06.09.17.23.18_veh-38_00773_01140.db
│     │    ├── ...
│     │    └── 2021.10.11.08.31.07_veh-50_01750_01948.db
│     └── trainval
│          ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
│          ├── 2021.06.09.17.23.18_veh-38_00773_01140.db
│          ├── ...
│          └── 2021.10.11.08.31.07_veh-50_01750_01948.db
└── sensor_blobs
        ├── 2021.05.12.22.00.38_veh-35_01008_01518
        │    ├── CAM_F0
        │    │     ├── c082c104b7ac5a71.jpg
        │    │     ├── af380db4b4ca5d63.jpg
        │    │     ├── ...
        │    │     └── 2270fccfb44858b3.jpg
        │    ├── CAM_B0
        │    ├── CAM_L0
        │    ├── CAM_L1
        │    ├── CAM_L2
        │    ├── CAM_R0
        │    ├── CAM_R1
        │    ├── CAM_R2
        │    └──MergedPointCloud
        │         ├── 03fafcf2c0865668.pcd
        │         ├── 5aee37ce29665f1b.pcd
        │         ├── ...
        │         └── 5fe65ef6a97f5caf.pcd
        │
        ├── 2021.06.09.17.23.18_veh-38_00773_01140
        ├── ...
        └── 2021.10.11.08.31.07_veh-50_01750_01948
```

### Create
In this part, we create a `json` file to read the data easily.
``` bash
python3 create_nuplan_json.py
```
Please remember to change `the path` in `create_nuplan_json.py`.


## Demo Data Preparation
In this section, we will explain how to compose our demo data and your own data. For our demo data, please directly download from [here](https://drive.google.com/file/d/1jJeBQKqRfy81aEPH4fSPxib0rM0m8rKQ/view?usp=drive_link).

The final data folder `DrivingWorld/data/` should be organized like this:

```
data
├── video-1
│   ├── 000000.png
│   ├── 000001.png
│   ├── ...
│   ├── 000015.png
│   ├── pose.npy
│   └── yaw.npy
├── video-2
│   ├── 000000.png
│   ├── 000001.png
│   ├── ...
│   ├── 000015.png
│   ├── pose.npy
│   └── yaw.npy
├── ...
├── video-n
│   ├── 000000.png
│   ├── 000001.png
│   ├── ...
│   ├── 000015.png
│   ├── pose.npy
│   └── yaw.npy
```
