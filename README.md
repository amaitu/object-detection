# object-detection

poetry install


```
poetry run python demo.py  \
        --prototxt MobileNetSSD_deploy.prototxt.txt \
        --model MobileNetSSD_deploy.caffemodel \
      --confidence .5
```