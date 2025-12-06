The ROOT files were too big to put into this repository; you may find them here: https://drive.google.com/drive/folders/1fnV6p6-O6p3FxKSw02DDOrfrZtwN_4IW?usp=drive_link

We need to create a conda environment with the packages listed in `yujina_versions.txt` and install pypi packages listed in `yujina_pypi_requirements`

Then, we can simply run `dropout.py` to train the model. It allows for various arguments to help select hyperparameters. Here are some example command lines:

Runs with with no hidden layers, so this is just logistic regression and we can treat it as the baseline
```
$ python dropout.py --output_dir output/baseline_logistic_regression_no_hidden_layers
```

Runs with 3 hidden layers of size 64 and no dropout
```
$ python dropout.py --hidden_dims 64 64 64 --output_dir output/hidden_dims_64_64_64
```

Runs with 3 hidden layers of size 64, dropout value of 0.5.
```
$ python dropout.py --hidden_dims 64 64 64 --dropout 0.5 --output_dir output/hidden_dims_64_64_64_dropout_0p5
```

Runs with 1 hidden layer of size 64 and LeakyReLU activation
```
$ python dropout.py --hidden_dims 64 --activation leaky_relu --output_dir output/hidden_dims_64_leaky_relu
```