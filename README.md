# KeywordSpotting4ESP32
## Keyword Spotting on Microcontrollers
Time: Feb 2024 - May 2024

This project aims to:
- Build a production device that continuously listens for users’ wake-up words, commands, first step of an arbitrary Virtual Assistant.
- Apply simple Deep learning neural networks to achieve high accuracy, while maintaining small footprint, efficient and low-energy consumption.
- Deploy to popular microcontrollers, such as ESP-based, and ARM-based.
- Propose a novel model that satisfies all the above criteria and surpasses other previous SOTA models

Benchmark dataset was used is Google Speech Commands v2, which can be found here: <https://arxiv.org/pdf/1804.03209>

Tools:
- Pytorch -> Research and experiment
- TensorFlow, TensorFlow Lite -> Deployment
- Arduino IDE -> Compiler

This project mostly based on the example micro_speech provided by TensorFlow, which can be found here: <https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech>

Here is some Demo Videos if you interest: <https://drive.google.com/drive/folders/1-3ld8DmlK0q0sXuZQUgdvyo8z4V3zNo9?usp=drive_link>

Each descriptions were made in respective folders, you can follow them to explore. If you have any question, just email me <thuan.tn210823@sis.hust.edu.vn> or <tranthuan10x@gmail.com>. 

References:
- Zhang, Yundong, et al. "Hello edge: Keyword spotting on microcontrollers." arXiv preprint arXiv:1711.07128 (2017).
- López-Espejo, Iván, et al. "Deep spoken keyword spotting: An overview." IEEE Access 10 (2021): 4169-4199.
- Hou, Jingyong, Lei Xie, and Shilei Zhang. "Two-stage streaming keyword detection and localization with multi-scale depthwise temporal convolution." Neural Networks 150 (2022): 28-42.
- Wei, Yungen, et al. "EdgeCRNN: an edge-computing oriented model of acoustic feature enhancement for keyword spotting." Journal of Ambient Intelligence and Humanized Computing (2022): 1-11.
- Kim, Byeonggeun, et al. "Broadcasted residual learning for efficient keyword spotting." arXiv preprint arXiv:2106.04140 (2021).
- Wang, Jie, et al. "WeKws: A production first small-footprint end-to-end Keyword Spotting Toolkit." ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023.
- Warden, Pete. "Speech commands: A dataset for limited-vocabulary speech recognition." arXiv preprint arXiv:1804.03209 (2018).

