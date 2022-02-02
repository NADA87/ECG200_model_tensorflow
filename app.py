# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser. #http://127.0.0.1:8050/

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from joblib import load

app = dash.Dash(__name__)




report = load('./report.joblib')
fig_report = go.Figure(data=[go.Table(
    header=dict(values=list(report.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=report.transpose().values.tolist(),
               fill_color='lavender',
               align='left'))
])

result = load('./model_result.joblib')
history = pd.DataFrame(result['model_history'])
history['epoch'] = [i for i in range(1,history.shape[0]+1)]
fig_history =px.line(history, x=history.epoch, y=['loss', 'precision','accuracy','val_loss','val_precision','val_accuracy'])


app.layout = html.Div(children=[
    html.H1(children='Model Evaluation'),

    html.H5(children=' Classification report'),
    dcc.Graph(
        id='classification_report',
        figure=fig_report
    ),

    html.H5(children='parameters vs epoch'),
    dcc.Graph(
        id='model history',
        figure=fig_history
    )





])

if __name__ == '__main__':
    app.run_server(debug=True,host='0.0.0.0')
