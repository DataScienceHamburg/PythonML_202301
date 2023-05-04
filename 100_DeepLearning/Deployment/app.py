#%% packages
from flask import Flask, request
from model_class import MultiClassNet
import torch
import requests
import json

#%% create model instance & load state dicts
model = MultiClassNet(HIDDEN_FEATURES = 6, NUM_CLASSES = 3, NUM_FEATURES = 4)
local_file_path_weights_biases = 'model_iris_state.pt'
model.load_state_dict(torch.load(local_file_path_weights_biases))

#%% model inference
# format: BS, var




#%%
app = Flask(__name__)

@app.route("/predict", methods = ['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return 'Please use POST method and pass data'
    
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        print(content_type)
        if (content_type == "application/json"):
            data_from_request = request.data.decode('utf-8')
            dict_data = json.loads(data_from_request.replace("'", "\""))
            data = dict_data['data']
            # data = [6.2, 3.4, 5.4, 2.3]
            X = torch.tensor([data])
            print(X)
            y_test_pred_softmax = model(X)
            y_test_pred_cls = torch.max(y_test_pred_softmax, 1).indices.numpy()[0]
            y_test_pred_cls

            cls_dict = {
                0: 'setosa', 
                1: 'versicolor',
                2: 'virginica'
            }

            result = f"Your flower belongs to class {cls_dict[y_test_pred_cls]}"
        return result

if __name__ == '__main__':
    app.run()