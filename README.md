# KeywordSpoting for Microcontrollers
This repository reimplements several state-of-the-art (SOTA) architectures for Keyword Spotting (KWS) using different approaches, including TCN, CRNN, and CNN, with the PyTorch framework. The models include **MDTC** from the paper '*Two-Stage Streaming Keyword Detection and Localization with Multi-Scale Depthwise Temporal Convolution*', **EdgeCRNN** from '*EdgeCRNN: An Edge-Computing Oriented Model for Acoustic Feature Enhancement in Keyword Spotting*', and **BC-ResNet** from '*Broadcasted Residual Learning for Efficient Keyword Spotting*'.

I have also proposed a novel model, **TF-ResNet**, which is highly competitive with the top-1 **BC-ResNet**.

The best models were deployed to ESP32 microcontrollers for real-world performance testing.
## Installation
Clone my repo
```bash
$ git clone https://github.com/thuantn210823/KeywordSpotting4ESP32.git
```
Install all required libraries in the `requirements.txt` file.
```bash
cd KeywordSpotting4ESP32
pip install -r requirements.txt
```
## Run
For training
```sh
cd KeywordSpotting4ESP32
py train.py --config_yaml YAML_PATH
```
For inference
```sh
cd KeywordSpotting4ESP32
py infer.py --config_yaml YAML_PATH --audio_path AUDIO_PATH
```
`Note:` If the above command doesn’t work, try replacing `py` with `python`, or the full `python.exe` path (i.e. `~/Python3xx/python.exe`) if the above code doesn't work.
## Example
```sh
cd KeywordSpotting4ESP32
py train.py --config_yaml conf/BCResNet/train.yaml
```
```sh
cd KeywordSpotting4ESP32
py infer.py --config_yaml conf/BCResNet/infer.yaml --audio_path example/right.wav
```
`Note:` Some arguments in these `train.yaml` files are still left blank waiting for you to complete. 

Here is what you should get for the inference run above:
```
Detected: right command!
```
## Pretrained Models
Pretrained models are offerred here, which you can find in the `pretrained` directory. All model checkpoints are available, except for MDTC, which I lost due to a failure to save it

## Results
I performed data augmentation offline, so the results may be slightly less accurate. The table below presents my results on the Benchmark dataset, Google Speech Commands version 2
| Model |#Param|#Mult| #Acc | #FAr| #FRr (keyword `on`)|
|:-:|:-:|:-:|:-:|:-:|:-:|
|MDTC 4-64-5|164K|12.27M|-|-|-|
|EdgeCRNN-1x|454.86K|14.04M|96.31%|0%|3.79%|
|BCResNet-3|54.2K|14.53M|98.09%|0%|3.28%|

## Novel Model
I have proposed a new architecture called TF-ResNet, along with its pretrained model, which is available here. Below are its architecture details and results. My baseline for comparison is BC-ResNet, with the Top-1 accuracy on the Google Speech Commands 12 dataset.
![TF-ResNet](https://github.com/thuantn210823/KeywordSpotting4ESP32/blob/main/fig/TFResNet_v1.png)
| Model |#Param|#Mult| #Acc |
|:-:|:-:|:-:|:-:|
|BCResNet-3|54.2K|**14.53M**|98.09%|
|TFResNetv1-3|**45.1K**|24.82M|97.98%|
|TFResNetv2-3|46.2K|**12.65M**|97.91%|

I can only share version 1 with you, which offers the same performance, fewer parameters, but slightly more operations. In version 2, I’ve addressed both issues. If you're interested, feel free to contact me for more information.

## Deployment
To evaluate performance, I deployed some of my models to the ESP32 microcontroller using `ONNX` and `TFLite`. For more details, you can check the `conversion.py` file and `Firmware` directory.
![Hardware](https://github.com/thuantn210823/KeywordSpotting4ESP32/blob/main/Hardware_ESP32_INMP441.jpg)

## Citation
Cite their great papers!
```
@article{zhang2017hello,
  title={Hello edge: Keyword spotting on microcontrollers},
  author={Zhang, Yundong and Suda, Naveen and Lai, Liangzhen and Chandra, Vikas},
  journal={arXiv preprint arXiv:1711.07128},
  year={2017}
}
```
```
@article{lopez2021deep,
  title={Deep spoken keyword spotting: An overview},
  author={L{\'o}pez-Espejo, Iv{\'a}n and Tan, Zheng-Hua and Hansen, John HL and Jensen, Jesper},
  journal={IEEE Access},
  volume={10},
  pages={4169--4199},
  year={2021},
  publisher={IEEE}
}
```
```
@article{wei2022edgecrnn,
  title={EdgeCRNN: an edge-computing oriented model of acoustic feature enhancement for keyword spotting},
  author={Wei, Yungen and Gong, Zheng and Yang, Shunzhi and Ye, Kai and Wen, Yamin},
  journal={Journal of Ambient Intelligence and Humanized Computing},
  pages={1--11},
  year={2022},
  publisher={Springer}
}
```
```
@article{kim2021broadcasted,
  title={Broadcasted residual learning for efficient keyword spotting},
  author={Kim, Byeonggeun and Chang, Simyung and Lee, Jinkyu and Sung, Dooyong},
  journal={arXiv preprint arXiv:2106.04140},
  year={2021}
}
```
```
@inproceedings{wang2023wekws,
  title={Wekws: A production first small-footprint end-to-end keyword spotting toolkit},
  author={Wang, Jie and Xu, Menglong and Hou, Jingyong and Zhang, Binbin and Zhang, Xiao-Lei and Xie, Lei and Pan, Fuping},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```
```
@article{warden2018speech,
  title={Speech commands: A dataset for limited-vocabulary speech recognition},
  author={Warden, Pete},
  journal={arXiv preprint arXiv:1804.03209},
  year={2018}
}
```
