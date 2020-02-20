# BootstrapNLP
BootstrapNLP is a Python-based web server that will serve an easy to interact
with webpage which will push and pull directly to a pre-trained tensorflow
model.

The intent of this application is to enable ML developers to quickly get a freshly
trained model into a UI that can be used for testing or demonstration purposes.
There are no fancy pipelines or architectures to learn. You can be up and running
in a couple of minutes.

The application is primarily configuration driven. The web page it serves can
host multiple models and allows you to configure multiple text inputs and outputs.

This project is still very much a work in progress, but here are some tips
in case you want to give it a try:

## Getting Started

You will bring:
1) A trained tensorflow model that you will serve using [tensorflow serving](https://www.tensorflow.org/tfx/serving/serving_basic)
2) An input and output processor which will take in text and produce inputs for
your model and take outputs and convert them into something pretty to view on
the webpage. Generally, the input processor is simply your tokenizer.

To get this thing running, here is what you will do:
1) Start up a tensorflow_serving server. Here is the command I use:
```
docker run -p 8500:8500 --mount type=bind,source=<saved_model_path>,target=<model_name_inside_saved_path> -e MODEL_NAME=<model_id, ex: 'gpt2'> -t tensorflow/serving
```
2) Edit model_configs.yaml from this repo. I've extensively documented this configuration
file. You will need to provide some basic information about your model, the inputs
it takes, the outputs it produces, and where the TF model server resides. Use the
values I provided as a guide.
3) Create your data processing script which was specified in the config. Generally the inputs
will just perform basic tokenization and the outputs will be fed directly through. See
processors/sentiment.py for an example.
4) Run flask_server.py and point your browser to http://localhost:8080

## Future Changes
I have a few focuses for this project at the moment:
1) Move from tensorflow serving to a model serving framework that supports pytorch
as well as tensorflow.
2) Retool HTML page to use Angular or React rather than a custom data binding
script.
3) Add additional input and output types as needed.

I will endeavor to only commit working code to GitHub. Pull requests are welcome.
