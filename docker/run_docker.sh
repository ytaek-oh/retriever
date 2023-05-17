CODEBASE_ROOT=$1
DATA_ROOT=$2

docker run --gpus all -it --shm-size=36gb \
    -v ${DATA_ROOT}:/home/appuser/datasets \
    -v ${CODEBASE_ROOT}:/home/appuser/retriever \
    --name retriever retriever:v0
