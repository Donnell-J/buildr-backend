FROM python:3.9-bullseye

WORKDIR tmp
COPY . /tmp

EXPOSE 9999
ENV TRANSFORMERS_CACHE /root/.cache/huggingface/hub

RUN pip install --upgrade pip
RUN pip install -r "requirements.txt" 
RUN  apt-get update && apt-get install libgl1-mesa-glx -y
ENTRYPOINT python main.py