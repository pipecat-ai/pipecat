FROM ubuntu:22.04

RUN apt-get update && apt-get install -y wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb

RUN echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list

RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get install -y cuda-nvcc-12-4 libcublas-12-4 libcudnn8

RUN mkdir /app
RUN mkdir /app/assets
RUN mkdir /app/utils
COPY *.py /app/
COPY requirements.txt /app/
copy assets/* /app/assets/
copy utils/* /app/utils/

WORKDIR /app
RUN pip3 install -r requirements.txt

EXPOSE 7860

CMD ["python3", "server.py"]
