# uses info from: https://github.com/jupyter/docker-stacks/blob/master/base-notebook/Dockerfile
FROM jupyter/scipy-notebook

ADD . /home/jovyan/hdfe
WORKDIR /home/jovyan/hdfe

# Install genelastpricing
USER root
RUN pip install -e .

EXPOSE 80

ENV NAME World

USER $NB_UID
WORKDIR $HOME
CMD ["start.sh", "jupyter lab"]
