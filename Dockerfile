FROM python:3

ADD . /home/jovyan/hdfe
WORKDIR /home/jovyan/hdfe

USER root
RUN python setup.py install
RUN pip install -r requirements-test.txt
CMD ["pytest"]
