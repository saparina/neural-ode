# Experiments with Neural Ordinary Differential Equations on image and text classification tasks

For image classification we use **ResNet** model and MNIST and CIFAR-10 datasets, while for text classifiacation we use **VdCNN**
model and Ag-News dataset.


## Requirments

* PyTorch >= 1.0
* NumPy

## Spiral experiment

Run ODE:

```
PYTHONPATH=. python ./experiments/spiral-torch.py
```
### Result

![spiral](https://github.com/saparina/neural-ode/blob/master/imgs/spiral.png)

## MNIST classification

Run **ResNet** with 6 blocks:
```
PYTHONPATH=. python ./experiments/train.py  --data mnist --save ./log_resnet6_mnist --save_every 50 \
--log_every 1 --optimizer sgd --lr 0.1 --num_res 6
```

Run **ResNet** with 1 block:
```
PYTHONPATH=. python ./experiments/train.py  --data mnist --save ./log_resnet1_mnist --save_every 50 \
--log_every 1 --optimizer sgd --lr 0.1 --num_res 1
```

Run **OdeNet** with explicit Runge-Kutta solver and tolerance 1e-2:
```
PYTHONPATH=. python ./experiments/train.py  --data mnist --save ./log_odenet_mnist --save_every 50 \
--log_every 1 --optimizer sgd --lr 0.1 --solver runge_kutta  --tol 1e-2 --use_ode
```
\* another possible option is explicit Euler solver: ```--solver euler ```

### Results
Test Accuracy           |  Loss
:-------------------------:|:-------------------------:
![mnist_score](https://github.com/saparina/neural-ode/blob/master/imgs/mnist_score.png) | ![mnist_loss](https://github.com/saparina/neural-ode/blob/master/imgs/mnist_loss.png)


| Model                | Test Error, % | #  parameters | Time (s/epoch) |
|----------------------|---------------|-----------------------|----------------|
| ResNet(6)              | 0.34          | 577 K                   | 13.18         |
| ResNet(1)              | 0.37       | 207 K                 | 11.21          |
| OdeNet (Runge-Kutta) |  0.45            |  207 K                     |         254.42        |

## CIFAR-10 classification

Run **ResNet** with 6 blocks:
```
PYTHONPATH=. python ./experiments/train.py  --data cifar --save ./log_resnet6_cifar --save_every 50 \
--log_every 1 --optimizer sgd --lr 0.1 --num_res 6
```

Run **ResNet** with 1 block:
```
PYTHONPATH=. python ./experiments/train.py  --data cifar --save ./log_resnet1_cifar --save_every 50 \
--log_every 1 --optimizer sgd --lr 0.1 --num_res 1
```

Run **OdeNet** with explicit Runge-Kutta solver and tolerance 1e-2 (may take a lot of time):
```
PYTHONPATH=. python ./experiments/train.py  --data cifar --save ./log_odenet_cifar --save_every 50 \
--log_every 1 --optimizer sgd --lr 0.1 --use_ode --solver runge_kutta  --tol 1e-2 
```


### Results
Test Accuracy           |  Loss
:-------------------------:|:-------------------------:
![cifar_score](https://github.com/saparina/neural-ode/blob/master/imgs/cifar_acc.png) | ![cifar_loss](https://github.com/saparina/neural-ode/blob/master/imgs/cifar_loss.png)



| Model                | Accuracy, % | #  parameters | Time (s/epoch) |
|----------------------|---------------|-----------------------|----------------|
| ResNet(6)              | 86.7          | 577 K                  | 12.25          |
| ResNet(1)              | 84.19         | 207 K                 | 9.84           |
| OdeNet (Runge-Kutta) |  84.85            |  207 K                     |         1860.31        |
| OdeNet (Euler)       | 84.62         | 207 K                 | 159.02         |

## Text classification

Download and create Ag-News data:

```
mkdir .data
mkdir .data/ag_news
cd .data/ag_news
wget https://raw.githubusercontent.com/tothanhtung0205/VDCNN/master/ag_news_csv/test.csv
wget https://raw.githubusercontent.com/tothanhtung0205/VDCNN/master/ag_news_csv/train.csv
echo -e 'World\nSports\nBusiness\nSci/Tech' > classes.txt
```

Run **VdCNN** with 6 blocks:
```
PYTHONPATH='.' python ./experiments/texts/vdcnn.py --batch_size 256 --max_epo 20 --save vdcnn6  
```

Run **VdCNN** with 1 block:
```
PYTHONPATH='.' python ./experiments/texts/vdcnn.py --batch_size 256 --max_epo 20 --save vdcnn1 \
--num_blocks 1
```

Run **OdeNet** with explicit Euler solver and tolerance 1e-2 (may take a lot of time):
```
PYTHONPATH='.' python ./experiments/texts/vdcnn.py --batch_size 256 --max_epo 20 --save vdcnn_ode \
--use_ode --solver euler  --tol 1e-2 
```


### Results
Test Accuracy           |  Loss
:-------------------------:|:-------------------------:
![text_score](https://github.com/saparina/neural-ode/blob/master/imgs/text_acc.png) | ![text_loss](https://github.com/saparina/neural-ode/blob/master/imgs/text_loss.png)



| Model                | Accuracy, % | #  parameters | Time (s/epoch) |
|----------------------|---------------|-----------------------|----------------|
| VdCNN(6)              | 88.46         | 287 K              | 311       |
| VdCNN(1)              | 87.75        | 162 K                 | 172           |
| OdeNet (Euler)       | 84.21       | 162 K                 |4874         |


## References

[Original implementation](https://github.com/rtqichen/torchdiffeq)

