FROM ubuntu:16.04

RUN apt-get update && apt-get install -y python3 python3-pip libgtk2.0-dev libgdcm-tools
RUN apt-get install -y python-gdcm

# Basic env setup
WORKDIR /opt


# Install requirements and module code
COPY requirements.txt /opt/requirements.txt
RUN python3.5 -m pip install --upgrade pip
RUN python3.5 -m pip install -r /opt/requirements.txt
COPY . /opt/

EXPOSE 8900
ENTRYPOINT [ "python3.5", "mock_server.py" ]
