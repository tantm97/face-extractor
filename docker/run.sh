chmod +x install.sh
chmod +x Dockerfile
docker build -t aidev-cuda11.5-cudnn8-devel-ubuntu18.04:1.0.0 .
sudo docker run -itd --name tandev --gpus all  --shm-size=2gb -p 9001:9001 --mount type=bind,source=/home/tan/workspace/git/face-extractor/,target=/opt aidev-cuda11.5-cudnn8-devel-ubuntu18.04:1.0.0
