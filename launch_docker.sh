docker run -it --rm --gpus all \
        --name airush \
        -v ${PWD}:/src \
        relilau/airush-2-4:latest \
        /bin/bash