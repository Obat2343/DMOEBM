# DMOEBM

## Checklist

- [x] Upload inference code
- [ ] Upload training code

***

## Install

```sh
git clone https://github.com/Obat2343/DMOEBM.git
mkdir git
cd git
```

Install following apps in git directory.

- Pyrep (<https://github.com/stepjam/PyRep>)
- CoppeliaSim (<https://www.coppeliarobotics.com/downloads.html>) # Please check the Pyrep repository to confirm the version of CoppeliaSim
- RLBench (<https://github.com/stepjam/RLBench>)
- RotationConinuity (<https://github.com/papagina/RotationContinuity>)

Next, Install requirements

```sh
pip install -r requirements.txt
```

***
## Dataset
To create the dataset for training and testing, please run the following command.

```sh
python create_dataset.py --task_list TaskA TaskB
```

***
## Download Pre-trained weights

```sh
mkdir result
cd result
```

Please download and unzip the file from https://drive.google.com/file/d/1ECP7Vsz7HkC7dbgYmnI7gG1zVZAlwaXM/view?usp=share_link

***
## Test

```sh
python Evaluate_EBMDMO_on_sim.py --EBM_path ../result/RLBench/PickUpCup/EBM_aug_frame_100_mode_6d_first_Transformer_vae_256_and_random_second_none_inf_sort/model/model_iter100000.pth --DMO_path ../result/RLBench/PickUpCup/DMO_iterative_5_frame_100_mode_6d_noise_Transformer_vae_256/model/model_iter100000.pth --tasks PickUpCup --inf_method_list DMO_keep
```
