# D-SRA-pytorch

This repository implements a deep-running model for super resolution of flow images captured by high-speed camera.
 Super resolution allows you to pass low resolution images to CNN and restore them to high resolution. 
 More details about D-SRA model can be found in following paper:
 [Deep-learning-based super resolution reconstruction of highspeed imaging in fluids] 
 
 ## architecture
 ![Degradation model] -------------------> contains all codes of degradation model, including:  train_De.py, predict_De.py
 [SRA model]  ----------------------->contains all codes of SRA model, including:  SRA_train.py, SRA _predict.py
 ![data] ---------------------> contains all data of D-SRA model training and testing, including LRHF, HRLF for Degradation model, LRHF and LRLF for SRA model
 

## how to train
run main file
#training of Degradation model:
```bash
python train_De.py   
```
#training of SRA model:
```bash
python SRA_train.py   
```


## how to predict
#testing of Degradation model:
```bash
python predict_De.py   
```
#testing of SRA model:
```bash
python SRA_predict.py   
```

Note: All the file path in original code have been repalced by 'your path', so you should enter you file path or filename before running the code. 
There is only partial data in the file, if you want to obtain all the data, please contact the author by email.
Due to copyright restrictions, it can only be used for research, not for commercial use.
