import flask
import yaml

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf
import numpy as np

app = flask.Flask(__name__)
configs = {}

MODEL_REQUEST_TIMEOUT_SECS = 5
channels = []
model_stubs = {}
processors = {}

def init():
    for name, model in configs["models"].items():
        channel = grpc.insecure_channel(model["serving_url"])
        model_stubs[name] = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        # Convert path into a package definition.
        processor_module = __import__(model["data_processing_script"], fromlist=["*"])
        processors[name] = processor_module.DataProcessor()

@app.route("/view/<path:filename>")
def get_file(filename):
    return flask.send_from_directory(configs["static_files_dir"], filename)

@app.route("/lm", methods=["get", "post"])
def list_models():
    return flask.jsonify(configs["models"])

@app.route("/predict", methods=["post"])
def predict():
    response = {"success": False}
    req_data = flask.request.get_json()
    if req_data is None:
        response["error"] = "No JSON request found."
        return flask.jsonify(response)
    if "model_id" not in req_data or req_data["model_id"] not in configs["models"]:
        response["error"] = "Model ID not specified or not found."
        return flask.jsonify(response)

    # Extract all inputs from the request and send them through the model processor.
    model_id = req_data["model_id"]
    model = configs["models"][model_id]
    inputs = {}
    for input_name, input_spec in model["inputs"].items():
        if input_name not in req_data:
            response["error"] = "Input %s not provided in request." % (input_name)
            return flask.jsonify(response)
        # TODO: Sanitize and check inputs. This is very insecure at the moment.
        inputs[input_name] = req_data[input_name]
    model_inputs = processors[model_id].process_input(inputs)
    model_inputs = np.reshape(model_inputs, (1, -1))

    # Perform the prediction.
    request = predict_pb2.PredictRequest()
    request.model_spec.name = configs["models"][model_id]['model_spec_name']
    request.inputs["input_ids"].CopyFrom(
        tf.make_tensor_proto(model_inputs, shape=[1, model_inputs[0].size]))
    # Block on response. Note that we could use future() to return immediately; not sure how useful that is though.
    model_response = model_stubs[model_id].Predict(request, MODEL_REQUEST_TIMEOUT_SECS)

    response["outputs"] = processors[model_id].process_output(model_response.outputs)
    response["success"] = True
    return flask.jsonify(response)

if __name__ == '__main__':
    with open('model_configs.yaml') as config_file:
        configs = yaml.full_load(config_file)
        init()
        app.run(debug=True, port=8080)