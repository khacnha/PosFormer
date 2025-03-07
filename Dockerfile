FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

# Tránh tương tác người dùng trong quá trình cài đặt
ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt các gói cần thiết
RUN apt-get update && apt-get install -y \
    python3.7 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pandoc \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép mã nguồn vào container
COPY . /app/

# Cài đặt các phụ thuộc Python
RUN pip3 install --no-cache-dir torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --no-cache-dir \
    pytorch-lightning==1.4.9 \
    torchmetrics==0.6.0 \
    pillow==8.4.0 \
    opencv-python \
    fastapi \
    uvicorn \
    python-multipart \
    jsonargparse[signatures] \
    einops


# Cài đặt torchvision từ nguồn cụ thể (nếu cần)
RUN pip3 install git+https://github.com/pytorch/vision.git@v0.2.2

# Mở cổng cho FastAPI
EXPOSE 8000

# Lệnh chạy khi container khởi động
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]