**327 Project - Comparing Pytorch and Tensorflow on their distributional performance**

Prerequisites:
Have a CPU similar to Intel core i7 new version and RAM >= 32GB

All these builds took ~15 mins on windows when using wsl with docker.
To speed up the training from each ~40 mins to each ~15 mins 
(not including downloading dataset and preparing data), remove the memory and cpu
flags in the commands and in the docker compose files. 

***Instructions to run:***
1. Clone the project
2. Have docker installed
3. Open Terminal
4. Go to project Folder
5. Execute Following commands:
   1. _For TF Single Worker:_
      1. docker build -t tf-single-worker -f TensorflowSingle.dockerfile .
      2. docker run -m 8g --cpus=2 tf-single-worker
   2. _For Pytorch Single worker:_
      1. docker build -t pytorch-single-worker -f PytorchSingle.dockerfile .
      2. docker run -m 8g --cpus=2 pytorch-single-worker
   3. _For TF Multi Worker:_
      1. docker build -t tf-multi-worker -f TensorflowMulti.dockerfile .
      2. docker compose up -f tf-docker-compose.yml
   4. _For Pytorch Multi Worker:_
      1. docker build -t pytorch-multi-worker -f PytorchMulti.dockerfile .
      2. docker run -m 8g --cpus=2 pytorch-multi-worker

