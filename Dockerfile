FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
ENV JAVA_HOME /usr/lib/jvm/java-1.7-openjdk/jre
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install gcc make git curl g++ default-jdk wget -y &&\
    apt-get autoremove -y  && \
    apt-get clean

RUN wget https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh
RUN chmod 755 script.deb.sh
RUN /bin/bash script.deb.sh
RUN apt-get install git-lfs
RUN git lfs install

ADD req.txt ./
RUN pip install --upgrade pip
RUN pip install -r req.txt