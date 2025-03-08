# MGTC

The source code of paper "MGTC: Multi-Granularity Temporal Aware Time Series Classification"

## Requirmenets:

- Python3.8
- Pytorch>=1.7
- Numpy
- Scikit-learn
- Pandas

## Datasets

### Download datasets

You can now find the preprocessed datasets [on this link](https://drive.google.com/drive/folders/1f1SQDdqUQ8g9YDmilb6zgyXkzw8dVeHs?usp=sharing)

## Training MGTC 

You can select one of two training modes:

 - Self-supervised training (self_supervised)
 - Fine-tuning the self-supervised model (fine_tune)

The code allows also setting a name for the experiment.
To use these options:

```
python main.py --training_mode self_supervised --selected_dataset LSST
```

Note that the name of the dataset should be the same name as inside the "data" folder, and the training modes should be
the same as the ones above.

To train the model for the `fine_tune` modes, you have to run `self_supervised` first.
