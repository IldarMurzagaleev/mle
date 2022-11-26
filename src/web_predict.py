from flask import Flask, render_template, request, jsonify
from src.predict import Predictor
from src.utils import vectorize

app = Flask(__name__)

@app.route('/', methods= ["GET", "POST"])
def hello_world():
    """
    Method for runing flask web-application with prediction of trained model
    """
    if request.method == 'POST':
        model = str(request.values.get('model'))
    else:
        model = "LOG_REG"
    test = 'func'
    pred_args = {'model': model, 'tests': test}
    predictor = Predictor()
    message=predictor.predict(pred_args)
    return message

if __name__ == '__main__':
    app.run(debug=True)
