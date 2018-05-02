# Train bidaf QA model

This document describes how to train bidaf QA model using allennlp.

## Setting up local development environment

### Setting up a virtual environment

[Conda](https://conda.io/) can be used set up a virtual environment
with the version of Python required for AllenNLP and in which you can
sandbox its dependencies. If you already have a python 3.6 environment
you want to use, you can skip to the 'installing via pip' section.

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Create a Conda environment with Python 3.6

    ```
    conda create -n allennlp python=3.6
    ```

3.  Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use AllenNLP.

    ```
    source activate allennlp
    ```
    
### Setting up a development environment

you'll need to install the library from GitHub and manually install the requirements:

1. First, clone the repo:

```
git clone https://github.com/ziyaoh/allennlp.git
```

2. Change your directory to where you cloned the files:

```
cd allennlp
```

3.  Install the required dependencies.

    ```
    INSTALL_TEST_REQUIREMENTS="true" ./scripts/install_requirements.sh
    ```

4. You'll also need to install PyTorch 0.3.1, following the appropriate instructions
for your platform from [their website](http://pytorch.org/).

You should now be able to test your installation with `./scripts/verify.py`.  Congratulations!

## Training a Model

In this tutorial we'll train a bidaf QA model using AllenNLP.
The model is defined in [allennlp/models/reading_comprehension/bidaf.py](https://github.com/ziyaoh/allennlp/blob/master/allennlp/models/reading_comprehension/bidaf.py).

One of the key design principles behind AllenNLP is that
you configure experiments using JSON files. (More specifically, [HOCON](https://github.com/typesafehub/config/blob/master/HOCON.md) files.)

Our bidaf configuration is defined in
[training_config/bidaf.json](https://github.com/ziyaoh/allennlp/blob/master/training_config/bidaf.json).
You can peek at it now if you want; we'll go through it in detail in the next tutorial.
Right at this instant you might care about the `trainer` section, which specifies how we want to train our model:

```js
"trainer": {
    "num_epochs": 20,
    "grad_norm": 5.0,
    "patience": 10,
    "validation_metric": "+em",
    "cuda_device": 0,
    "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "max",
        "patience": 2
    },
    "optimizer": {
        "type": "adam",
        "betas": [0.9, 0.9]
    }
}
```

Here the `num_epochs` parameter specifies that we want to make 20 training passes through the training dataset. `patience`
controls the early stopping -- if our validation metric doesn't improve for
this many epochs, training halts. And if you have a GPU you can make `cuda_device` to 0 to use it. Otherwise, change it to -1.

Change any of those if you want to, and then run

```
$ allennlp train training_config/bidaf.json --serialization-dir /tmp/allennlp/bidaf
```

The `serialization-dir` argument specifies the directory where the model's vocabulary and checkpointed weights will be saved.

This command will download the datasets and cache them locally,
log all of the parameters it's using,
and then display the progress and results of each epoch.
You can also manually download the dataset to local machine and then specify the path to your dataset in the configuration json file.

```js
"train_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-train-v1.1.json",
"validation_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-dev-v1.1.json",
```

Now that the model is trained, there should be a bunch of files in the serialization directory. The `vocabulary` directory
contains the model's vocabularies, each of which is a (distinct) encoding of strings as integers.
In our case, we'll have one for `tokens` (i.e. words) and another for `tags`. The various
`training_state_epoch_XX.th` files contain the state of the trainer after each epoch (`.th` is the suffix for serialized torch tensors),
so that you could resume training where you left off, if you wanted to.
Similarly, the `model_state_epoch_XX.th` files contain the model weights after each epoch.
`best.th` contains the *best* weights (that is, those from the epoch with the smallest `loss` on the validation dataset).

Finally, there is an "archive" file `model.tar.gz` that contains the training configuration,
the `best` weights, and the `vocabulary`.

## Evaluating a Model

Once you've trained a model, you likely want to evaluate it on another dataset.

We can use the `evaluate` command, giving it the archived model and the evaluation dataset:

```
$ allennlp evaluate /tmp/allennlp/bidaf/model.tar.gz --evaluation-data-file https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-dev-v1.1.json
```

When you run this it will load the archived model, download and cache the evaluation dataset, and then make predictions:

There is also a command line option to use a GPU, if you have one.

## Making Predictions

Finally, what's the good of training a model if you can't use it to make predictions?
The `predict` command takes an archived model and a [JSON lines](https://en.wikipedia.org/wiki/JSON_Streaming#Line_delimited_JSON)
file of inputs and makes predictions using the model.

Here, the "predictor" for the tagging model expects a JSON blob containing a sentence:

```bash
$ cat <<EOF >> inputs.txt
{"sentence": "I am reading a tutorial."}
{"sentence": "Natural language processing is easy."}
EOF
```

After which we can make predictions:

```bash
$ allennlp predict /tmp/tutorials/getting_started/model.tar.gz inputs.txt
... lots of logging omitted
{"tags": ["ppss", "bem", "vbg", "at", "nn", "."], "class_probabilities": [[ ... ]]}
{"tags": ["jj", "nn", "nn", "bez", "jj", "."], "class_probabilities": [[ ... ]]}
```

Here the `"tags"` are the part-of-speech tags for each sentence, and the
`"class_probabilities"` are the predicted distributions of tags for each sentence
(and are not shown above, as there are a lot of them).
