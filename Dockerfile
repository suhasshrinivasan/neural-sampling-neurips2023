FROM walkerlab/pytorch-jupyter:cuda-11.7.1-pytorch-1.13.1-torchvision-0.13.0-torchaudio-0.11.0-ubuntu-20.04

RUN python3 -m pip install --upgrade pip

RUN pip3 install scikit-image wandb 

RUN git clone -b neural-sampling-neurips2023 https://github.com/suhasshrinivasan/neuralpredictors.git /src/neuralpredictors &&\
    pip3 install /src/neuralpredictors &&\
    git clone -b neural-sampling-neurips2023 https://github.com/sinzlab/nnsysident.git /src/nnsysident &&\
    pip3 install /src/nnsysident &&\
    git clone -b neural-sampling-neurips2023 https://github.com/sinzlab/nnvision.git /src/nnvision &&\
    pip3 install /src/nnvision &&\
    git clone -b neural-sampling-neurips2023 https://github.com/suhasshrinivasan/gensn.git /src/gensn &&\
    pip3 install /src/gensn

COPY . /src/project
RUN pip3 install -e /src/project

ENTRYPOINT ["bash", "-c"]