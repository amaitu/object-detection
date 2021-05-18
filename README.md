# object-detection

## Installation
Install poetry - https://python-poetry.org/docs/

```
poetry install
```

## Running

### Camera Test
```
poetry run python camera_test.py -p=1
```

### Object detection
```
poetry run python object_detection.py  \
        --prototxt models/MobileNetSSD_deploy.prototxt.txt \
        --model models/MobileNetSSD_deploy.caffemodel \
      --confidence .5
```

### Colour detection
```
poetry run python color_detection.py --colour=red
```


## Definitions

### color space
Method for grouping colours - often implies the use of a colour model, e.g. RGB.

