__author__ = "Andrei Ermishin"
__license__ = "GNU GPLv3"
__email__ = "andrey.yermishin@gmail.com"


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from flask import Flask, request, jsonify, abort


def make_clusters(table, num_clusters=5):
    """
    Return a Pandas DataFrame with assigned clusters as new column.
    table is a JSON-like list.
    """
    data = pd.DataFrame(table)

    # X = data.drop('Персона', axis='columns')
    X = data.drop(data.columns[0], axis='columns')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    agglo = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    agglo.fit(X_scaled)

    data[f'{num_clusters}кластеров'] = agglo.labels_

    return data


app = Flask(__name__)


@app.route('/')
def homepage():
    return '<h2>Greetings from the service!</h2>'


@app.route('/api/v1.0/cluster', methods=['GET', 'POST'])
def cluster():
    """ Read JSON data (as records) and return data with clusters as JSON. """
    data_5columns = request.get_json()  # return list here
    if not data_5columns:
        abort(400)
    
    return make_clusters(data_5columns).to_json(orient='records')


if __name__ == '__main__':
    app.run(debug=True)


# test command:
# curl --header "Content-Type: application/json" --data "{\"a\":5,\"b\":12}" http://127.0.0.1:5000//api/v1.0/cluster

# commands:
# curl --header "Content-Type: application/json" --data @table.json http://127.0.0.1:5000//api/v1.0/cluster

# curl --header "Content-Type: application/json" --data @table.json http://127.0.0.1:5000//api/v1.0/cluster --output result.json
