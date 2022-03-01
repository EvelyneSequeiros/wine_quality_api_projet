import joblib
from flask import Flask, request, jsonify, render_template

MODEL_PATH = "resources/model.joblib"

app = Flask(__name__, template_folder='.')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])    
def predict():
    if request.json:
        json_input = request.get_json()
        
        #Load model
        classifier = joblib.load(MODEL_PATH)
        
        #Compute predictions
        predictions = []
        for i in range(len(json_input["input"])) :
            prediction= classifier.predict([json_input["input"][i]])
            #prediction = float(prediction[0])
            predictions.append(float(prediction[0]))
        
        # Return predictions
        response = {
            "prediction" : predictions,
        }
        return jsonify(response), 200       
    return jsonify({"msg": "Error, no JSON detected"})



if __name__ == "__main__":
    app.run(debug=True)