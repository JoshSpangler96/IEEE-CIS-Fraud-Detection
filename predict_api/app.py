from flask import Flask
from flask_restful import Api, Resource, reqparse
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
api = Api(app)


class Predict(Resource):

    @staticmethod
    @app.route('/predict', methods=["POST"])
    def post():
        model = pickle.load(open('../model/lgb_model.pkl', 'rb'))
        parser = reqparse.RequestParser()
        parser.add_argument('identity_path')
        parser.add_argument('transaction_path')

        args = parser.parse_args()
        df = pipelines.ieee_test_pipeline(
            identity_path=args['identity_path'],
            transaction_path=args['transaction_path']
        )
        logging.info('Predicting Test Data')
        output = model.predict(df)
        output = dict(enumerate(output.flatten(), 1))
        logging.info('Finished Predicting Test Data')
        return output, 200


class Train(Resource):

    @staticmethod
    @app.route('/train', methods=["POST"])
    def train_model():
        parser = reqparse.RequestParser()
        parser.add_argument('identity_path')
        parser.add_argument('transaction_path')
        args = parser.parse_args()
        lgb_model = pipelines.ieee_train_pipeline(
            identity_path=args['identity_path'],
            transaction_path=args['transaction_path']
        )
        output = {'Model': str(lgb_model)}
        return output, 200


api.add_resource(Predict, '/predict')
api.add_resource(Train, '/train')

if __name__ == '__main__':
    app.run(debug=True, port='1080')

