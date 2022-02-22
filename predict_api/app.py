from flask import Flask, request, jsonify, render_template
from io import StringIO
import pandas as pd
import pickle
import pipelines
import logging
import sys

logging.basicConfig(
    format='[%(asctime)s|%(module)s.py|%(levelname)s]  %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout
)

app = Flask(__name__)


@app.route('/')
def description():
    return 'Flask inside Docker'


@app.route('/predict_demo', methods=["POST"])
def predict_demo():
    model = pickle.load(open('/app/model/lgb_model.pkl', 'rb'))
    df_id = pd.read_csv('/app/demo_data/demo_identity.csv')
    df_tran = pd.read_csv('/app/demo_data/demo_transaction.csv')
    df = pipelines.ieee_test_pipeline(
        identity=df_id,
        transaction=df_tran
    )
    logging.info('Predicting Test Data')
    output = model.predict(df)
    output = dict(enumerate(output.flatten(), 1))
    logging.info('Finished Predicting Test Data')
    return jsonify(output)


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/uploader', methods=[ 'GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f_id = request.files['id']
        df_id = pd.read_csv(StringIO(str(f_id.read(),'utf-8')))
        # f_id.save(f_id.filename)
        f_tran = request.files['tran']
        df_tran = pd.read_csv(StringIO(str(f_tran.read(),'utf-8')))
        # f_tran.save(f_tran.filename)
        df = pipelines.ieee_test_pipeline(
                identity=df_id,
                transaction=df_tran
        )
        logging.info('Predicting Test Data')
        model = pickle.load(open('/app/model/lgb_model.pkl', 'rb'))
        output = model.predict(df)
        output = dict(enumerate(output.flatten(), 1))
        logging.info('Finished Predicting Test Data')
        return 'finished'

# @app.route('/predict', methods=["POST"])
# def predict():
#     model = pickle.load(open('app/model/lgb_model.pkl', 'rb'))
#     parser = reqparse.RequestParser()
#     parser.add_argument('identity_path')
#     parser.add_argument('transaction_path')
#     args = parser.parse_args()
#
#     df = pipelines.ieee_test_pipeline(
#         identity_path=args['identity_path'],
#         transaction_path=args['transaction_path']
#     )
#     logging.info('Predicting Test Data')
#     output = model.predict(df)
#     output = dict(enumerate(output.flatten(), 1))
#     logging.info('Finished Predicting Test Data')
#     return jsonify(output)


# @app.route('/train', methods=["POST"])
# def train():
#     parser = reqparse.RequestParser()
#     parser.add_argument('identity_path')
#     parser.add_argument('transaction_path')
#     args = parser.parse_args()
#     lgb_model = pipelines.ieee_train_pipeline(
#         identity_path=args['identity_path'],
#         transaction_path=args['transaction_path']
#     )
#     output = {'Model': str(lgb_model)}
#     return output, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1080, debug=True)


