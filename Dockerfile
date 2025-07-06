FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg git && \
    pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY . .

CMD ["python", "dispatcher.py"]