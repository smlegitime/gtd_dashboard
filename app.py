import flask
import os

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from display_components import SIDEBAR_STYLE, SIDEBAR_HIDEN, CONTENT_STYLE, CONTENT_STYLE1
from display_components import create_card, create_navbar
from utils import RENAMED_COLUMNS
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import dash_cytoscape as cyto
import dash_daq as daq
from collections import defaultdict, Counter
import plotly.figure_factory as ff

# Dataframes
terror_df = pd.read_csv("data/globalterrorismdb_0718dist.csv", encoding ='latin1')
nltk.download('stopwords')

# Renaming some features and selecting those we want
terror_df.rename(columns=RENAMED_COLUMNS,inplace=True)
terror_df = terror_df[['Year','Month','Day','Country','state','Region','city','latitude','longitude','Attack_type','AttackType',
                      'Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive', 'PropertyDamage', 
                      'PropertyDamageExtent', 'success', 'WeaponType']]

# Preprocessing to create dates
terror_df['Day'] = terror_df['Day'].replace(0,15)
terror_df['Month'] = terror_df['Month'].replace(0,6)
terror_df['Date'] = pd.to_datetime(terror_df[['Year', 'Month', 'Day']])

# Adding casualties field casualties = killed + wounded
terror_df['Wounded'] = terror_df['Wounded'].fillna(0).astype(int)
terror_df['Killed'] = terror_df['Killed'].fillna(0).astype(int)
terror_df['casualties'] = terror_df['Killed'] + terror_df['Wounded']

print(len(terror_df.columns))

incidents_per_year = pd.DataFrame(terror_df['Year'].value_counts())
incidents_per_year.reset_index(level=0, inplace=True)
incidents_per_year = incidents_per_year.rename(columns={'index':'Year', 'Year':'incidents'})
incidents_per_year = incidents_per_year.sort_values(by='Year').reset_index(drop=True)

countries_by_decade = terror_df[terror_df['Year'].between(1970, 1980, inclusive='both')]

terror_copy = countries_by_decade.sort_values(by='casualties',ascending=False)[:30]


casualties_per_year= terror_df[['Year','casualties']].groupby(['Year'],axis=0).sum().sort_values('Year')
casualties_per_year = casualties_per_year.reset_index()

incidents_per_year['casualties'] = casualties_per_year['casualties']


# Natural Language Processing
stop = set(stopwords.words('english'))

# ****** APP LOGIC ******
load_figure_template("solar")

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SOLAR, dbc.icons.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Global Terrorism Dashboard (1970 - 2017)"
server = app.server

@server.route('/static/<path:path>')
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(os.path.join(root_dir, 'static'), path)

app.config["suppress_callback_exceptions"] = True

# **** UTILS *****
def make_transparent(graph):
    graph.update_layout({
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'font_color': '#FAF9F6'
})

def get_most_popular(df, attr):
    return df[attr].mode().item()

# ******* DEFAULT GRAPHS EXPLORE *******
#map
def_map = px.density_mapbox(terror_df, lat="latitude", lon="longitude", 
            z='casualties', animation_frame='Year', hover_data=['Date', 'Country', 'city'],
            radius=10, zoom=1.25, height=500)
def_map.update_layout(mapbox_style="stamen-terrain", margin={"r":0,"t":0,"l":0,"b":0})
make_transparent(def_map)

# ******* DEFAULT GRAPHS ANALYZE *******


chi_square_options = ['Region', 'PropertyDamageExtent', 'AttackType', 'Target_type', 'Weapon_type']
predict_options = ['Killed', 'Wounded', 'Attack_type', 'WeaponType']

def_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'label': 'data(label)',
            'color':'#fff'
        }
        
    },
    {"selector": "edge", "style": {"width": 1,}},
]

#************ DASH APP STUFF ***********
decades = [1970, 1980, 1990, 2000, 2010]

sidebar = html.Div(
        [
            # html.H4("Sub Menu", style={"color": "#FAF9F6"}, className="display-8"),
            html.Div([
                html.H5('Video',style={"margin-top": "0","font-weight": "bold","text-align": "center", "color": "#FAF9F6"}),
                dbc.Nav(
                    [
                        dbc.NavLink("Link to Video", href="/video", id="video-link", active="exact"),
                    ],
                    vertical=True,
                    pills=True,
                ),
                html.Hr(),
                html.H5('Menu',style={"margin-top": "0","font-weight": "bold","text-align": "center", "color": "#FAF9F6"}),
                html.H6('Region',style={"margin-top": "0","font-weight": "bold", "color": "#FAF9F6"}),
                daq.BooleanSwitch(
                    id='all-regions',
                    label='All',
                    on=True
                ),
                html.Br(),
                html.Div(id='r-dropdown', children=[
                    dcc.Dropdown(
                        id='regions-dropdown',
                        options=[{'value': num, 'label': num} for num in terror_df['Region'].unique().tolist()],
                        placeholder="Select a region",
                        multi=True
                    )
                ], style={'display':'none'}),
            ]),
            
            # html.Div(id='c-dropdown'),
            html.Div([
                html.Hr(),
                html.H6('Decade',style={"margin-top": "0","font-weight": "bold", "color": "#FAF9F6"}),
                dcc.Checklist(
                    id='decade',
                    options=[{'value': num, 'label': '{}s'.format(num)} for num in decades],
                    value=[2000, 2010],
                    labelStyle={'display': 'inline-block','padding': '2px', 'margin':'10px', "color": "#FAF9F6"}
                ),
            ]),
            
            html.Hr(),
            
        ],
        id="sidebar",
        style=SIDEBAR_STYLE,
    )

# Row
row_explore = html.Div(
    [
        html.H3('Explore',style={"color": "#FAF9F6"},),
        dbc.Spinner(dbc.Row(
            [
                dbc.Col(html.Div(
                    children=[
                        dcc.Graph(id="geo-map", figure=def_map)
                    ], 
                    className="p-3 bg-dark rounded-3",),loading_state=True)
            ],className="p-3 bg-default rounded-3"
        ,style={"color": "#FAF9F6"},)),
        html.Br(),
        dbc.Spinner(html.Div(id='trend-cards')),
        html.Br(),
        dbc.Spinner(dbc.Row(
            [
                dbc.Col(html.Div(children=[
                    dcc.Graph(id='treemap')
                ], className="p-3 bg-dark rounded-3",), md=5),
                dbc.Col(html.Div(children=[
                    dcc.Graph(id='time-series')
                ], className="p-3 bg-dark rounded-3",), md=7)
            ],className="p-3 bg-default rounded-3"
        ,style={"color": "#FAF9F6"},)),
        html.Br(),
        dbc.Spinner(dbc.Row(
            [
                dbc.Col(html.Div(children=[
                    dcc.Graph(id='bar-chart-explore')
                ], className="p-3 bg-dark rounded-3",), md=6),
                dbc.Col(html.Div(children=[
                    dcc.Graph(id='pie-chart')
                ], className="p-3 bg-dark rounded-3",), md=6)
            ],className="p-3 bg-default rounded-3"
        ,style={"color": "#FAF9F6"},)),
        html.Br(),
        html.Footer('Created by: Sybille Légitime',style={"color": "#FAF9F6"},)
    ]
)
row_analyze = html.Div(
    [
        html.H3('A Deeper Look',style={"color": "#FAF9F6"},),
        dbc.Spinner(dbc.Row(
            [
                html.H6("Gaining Insights from Clusters", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                html.Hr(),
                html.P("Let us attempt to find patterns in the data. K-means clustering will help us find possible groupings between the frequency of terrorist groups and casualties. "
                "On the right, we take the components parts of the 'casualties' feature (killed and wounded), and cluster them through a Gaussian Mixture Model with the number of terrorist groups. All data are standardized for ease of visualization.", style={"text-align": "justify"}),
                dbc.Col(html.Div(children=[
                    html.Div(
                        [
                            dbc.Label("Cluster count"),
                            dbc.Input(id="cluster-count", type="number", value=3),
                        ]
                    ),
                    dcc.Graph(id="cluster-graph")
                ], className="p-3 bg-dark rounded-3",), md=6),
                dbc.Col(html.Div(children=[
                    html.Div(
                        [
                            dbc.Label("Number of components"),
                            dbc.Input(id="cluster-count-gmm", type="number", value=2),
                        ]
                    ),
                    dcc.Graph(id="3d-scatter")
                ], className="p-3 bg-dark rounded-3",), md=6),
            ],className="p-3 bg-dark rounded-3"
        ,style={"color": "#FAF9F6"},)),
        html.Br(),
        dbc.Spinner(dbc.Row(
            [
                html.H6("Text Analysis", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                html.Hr(),
                html.P("Here, the aim is to analyze various incident summaries and generate insight as to which terms were used the most for events occurring in a specific region, or decade in time. Words with similar frequencies are linked. You can move th nodes and reposition the graph." 
                    , style={"text-align": "justify"}),
                html.P('Click on a node to reveal the bar graph', style={"font-weight": "bold",}),
                dbc.Col(html.Div(
                    children=[
                        cyto.Cytoscape(
                            id='cytoscape',
                            style={'width': '100%', 'height': '550px',},
                            stylesheet=def_stylesheet,
                            layout={
                                'name': 'cose'
                            }
                        )
                    ], 
                    className="p-3 bg-dark rounded-3"), md=6),
                    dbc.Col(
                    html.Div(id='text-bar-graph', children=[
                        dcc.Graph(id="text-bar"),
                    ], 
                    className="p-3 bg-dark rounded-3",
                    style={"display":"none"}), md=6),
            ],className="p-3 bg-dark rounded-3"
        ,style={"color": "#FAF9F6"},)),
        html.Br(),
        dbc.Spinner(dbc.Row(
            [
                html.H6("Distibutions Analysis", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                html.Hr(),
                html.P("Let us take a look at the distributions of casualties to gain some further insight. The data on the histogram represent a distribution of the average number of "
                "casualties per incident (between 5 and 100). The data for the boxplots include all casualties The data are also normalized for ease of visualization.", style={"text-align": "justify"}),
                dbc.Col(
                    html.Div([
                        dcc.Graph(id="hist-graph"),
                    ], 
                    className="p-3 bg-dark rounded-3"), md=6),
                dbc.Col(html.Div(
                    children=[
                        dcc.Graph(id="boxplot"),
                    ], 
                    className="p-3 bg-dark rounded-3"), md=6),
            ],className="p-3 bg-dark rounded-3"
        ,style={"color": "#FAF9F6"},)),
        html.Br(),
        dbc.Spinner(dbc.Row(
            [    
                html.H6("Correlation Analyses", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                html.Hr(),
                html.P("The magnitude and direction of the realtionships between our features are very useful in general understanding of the data, and with prediction tasks."
                " Pairwise correlations between the features, and a regression against time can help us shed a light on which of them are best linked with casualties.", style={"text-align": "justify"}),
                dbc.Col(html.Div(children=[
                    dcc.Graph(id='heatmap'),
                ], className="p-3 bg-dark rounded-3",), md=6),
                dbc.Col(html.Div(children=[
                    dcc.Graph(id='regression'),
                ], className="p-3 bg-dark rounded-3",), md=6),
            ],className="p-3 bg-dark rounded-3"
        ,style={"color": "#FAF9F6"},)),
        html.Br(),
        dbc.Spinner(dbc.Row(
            [
                html.H6("Hypothesis Testing", style={"margin-top": "0","font-weight": "bold","text-align": "center", "color": "#FAF9F6"}),
                html.Hr(),
                html.P("With correlation and regression, we tested relationships between our numerical variables. Now, to look at relationships between our categorical variables, we will perform"
                " a Chi-Square test for independence. The null hypothesis (H0) states that our categorical variables are independent."
                , style={"text-align": "justify", "color": "#FAF9F6"},),
                dbc.Row([
                    dbc.Col(html.Div(children=[
                        dcc.Dropdown(
                            id='ctrl-left',
                            options=[{"label": opt, "value": opt} for opt in chi_square_options],
                            value=chi_square_options[0]
                        ),
                    ], className="p-3 bg-dark rounded-3",)),
                    dbc.Col(html.P('against', style={"margin-top": "0","text-align": "center", "color": "#FAF9F6"}),  className="p-3 bg-dark rounded-3",),
                    dbc.Col([
                        dcc.Dropdown(
                            id='ctrl-right',
                            options=[{"label": opt, "value": opt} for opt in chi_square_options],
                            value=chi_square_options[1]
                        ),
                    ],  className="p-3 bg-dark rounded-3",),
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='contingency-table'), md=9),
                    dbc.Col(html.Div(id ='chisquare-card', className="p-3 bg-dark rounded-3",), align='center', md=3),
                ]),
            ],className="p-3 bg-dark rounded-3"
        ,)),
        html.Br(),
        html.Footer('Created by: Sybille Légitime',style={"color": "#FAF9F6"},)
    ]
)

row_predict = html.Div(
    [
        html.H3('Time to Make Predictions',style={"color": "#FAF9F6"},),
        dbc.Spinner(dbc.Row(
            [
                html.H6("Classify Future Events", style={"margin-top": "0","font-weight": "bold","text-align": "center", "color": "#FAF9F6"}),
                html.Hr(),
                html.P("Here, we will be a Decision Tree classifier and inspect its performance. Our dependent variable is 'success', and is defined by whether or not a terrorist strike was successful" 
                "This variable is pretty confusing, so we'll see which features have the highest influence over the predictions.", style={"text-align": "justify", "color": "#FAF9F6"}),
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(
                            id='ctrl-classifier',
                            options=[{'value': x, 'label': x} for x in predict_options],
                            placeholder="Select a feature",
                            value= ['Killed', 'Wounded', 'Attack_type' ],
                            multi=True
                        ),
                    ),
                    dbc.Col(
                        dbc.Button("Train", id='train-classifier', color="secondary", className="me-1"),
                    ),
                ]),
                html.Div(id='classifier-graph', children=[
                    dbc.Row([
                        dbc.Col(html.Div(children=[
                            dcc.Graph(id="decision-tree")
                        ], className="p-3 bg-dark rounded-3",), md=9),
                        dbc.Col(html.Div(id='tree-results'), md=3),
                    ]),
                ], style={"display":"none"}),
                
            ],className="p-3 bg-dark rounded-3"
        )),
        html.Br(),
        dbc.Spinner(dbc.Row(
            [
                html.H6("Use Bayesian Inference", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                html.Hr(),
                html.P("Lastly, we will do some inference using the Bayesian method. Since the data is the frequency of casualties, and casualties are a discrete value, " 
                "our prior and likelihood are follow a Poisson distribution. The range of casualties is set between 1 and 5", style={"text-align": "justify", "color": "#FAF9F6"}),
                dbc.Col(html.Div(
                    children=[
                        dbc.Row([
                            html.P("Prior Rate:", style={"text-align": "justify", "color": "#FAF9F6"}),
                            dcc.Input(id='lamda-prior', value= 1, type="number",),
                        ]),
                        html.Br(),
                        dbc.Row([
                            html.P("Candidate Rates:", style={"text-align": "justify", "color": "#FAF9F6"}),
                            dcc.Input(id='lamda', value= 2, type="number",),
                        ]),
                    ], 
                    className="p-3 bg-dark rounded-3",), md=3),
                dbc.Col(html.Div(
                    children=[
                        dcc.Graph(id="bayes")
                    ], 
                    className="p-3 bg-dark rounded-3",), md=9)
            ],className="p-3 bg-dark rounded-3"
        ,style={"color": "#FAF9F6"},)),
        html.Br(),
        html.Footer('Created by: Sybille Légitime',style={"color": "#FAF9F6"},)
    ]
)

row_video = html.Div(
    [
        html.H3('Video Presentation',style={"color": "#FAF9F6"},),
        dbc.Spinner(dbc.Row(
            [
                dbc.Row([
                    
                    html.Video(
                        controls = True,
                        id = 'movie_player',
                    ## creat a new folder called static and place your video inside the static folder
                        src = "/static/sml20014FinalDataVizPresentation.mp4",
                        autoPlay=True
                    ),
                    
                ]),
            ],className="p-3 bg-dark rounded-3"
        )),
        html.Footer('Created by: Sybille Légitime',style={"color": "#FAF9F6"},)
    ]
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Store(id='side_click'),
    dcc.Location(id='url'),
    create_navbar(),
    sidebar,
    content,
])

@app.callback(
    Output('r-dropdown', 'style'),
    [Input('all-regions', 'on')])
def display_regions_dropdown(on):
    if not on:
        return {'display':'block'}
    else:
        return {'display':'none'}

@app.callback(
    Output('regions-dropdown', 'value'),
    [Input('all-regions', 'on')])
def set_regions_options(on):
    if not on:
        return ['North America', 'Sub-Saharan Africa', 'South Asia']
    else:
        return terror_df['Region'].unique().tolist()

@app.callback(
    Output("trend-cards", "children"),
    [Input("decade", "value"),
    Input("regions-dropdown", "value"),],
)
def display_trend_cards(years, regions):
    df = terror_df[terror_df['Year'].isin(years)]
    if len(regions) > 0:
        df = df[df['Region'].isin(regions)]
    n_groups= pd.DataFrame(df['Group'].value_counts()).iloc[1:,:]
    n_groups = n_groups.reset_index()
    most_pop_group = n_groups['index'].loc[0]

    return dbc.Row(
            children=[
                dbc.Col(dbc.Card(
                    create_card('bi bi-pin-map-fill','  Region most affected by attacks', get_most_popular(df, 'Region'),),
                    color='dark', inverse=True), width=6, lg=3),
                dbc.Col(dbc.Card(
                    create_card('bi bi-people-fill','  Country most affected by attacks', get_most_popular(df, 'Country'),), 
                    color='dark', inverse=True), width=6, lg=3),
                dbc.Col(dbc.Card(
                    create_card('bi bi-flag','  Group with the most attacks', most_pop_group,), 
                    color='dark', inverse=True), width=6, lg=3),
                dbc.Col(dbc.Card(
                    create_card('bi bi-exclamation-octagon-fill','  Most used weapon type', get_most_popular(df, 'Weapon_type'),), 
                    color='dark', inverse=True), width=6, lg=3),
            ],
            className="p-3 bg-default rounded-3"
        ,style={"color": "#FAF9F6"},)

@app.callback(
    Output('treemap', 'figure'),
    [Input("decade", "value"),
    Input("regions-dropdown", "value"),]
)
def display_treemap(years, regions):
    df = terror_df[terror_df['Year'].isin(years)]
    if len(regions) > 0:
        df = df[df['Region'].isin(regions)]
    sub_df = pd.DataFrame(df[['Country', 'Region', 'Year', 'AttackType', 'Attack_type']].value_counts()).reset_index()
    sub_df = sub_df.rename(columns={0:'incidents'})
    treemap = px.treemap(sub_df, path=[px.Constant("world"), 'Region', 'Country'], values='incidents',
                    title='Terrorist Incidents Breakdown ')
    treemap.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    make_transparent(treemap)

    return treemap

@app.callback(
    Output('time-series', 'figure'),
    [Input("regions-dropdown", "value"),]
)
def update_time_series(regions):
    df = terror_df[terror_df['Region'].isin(regions)]
    incidents_per_year = pd.DataFrame(df['Year'].value_counts())
    incidents_per_year.reset_index(level=0, inplace=True)
    incidents_per_year = incidents_per_year.rename(columns={'index':'Year', 'Year':'incidents'})
    incidents_per_year = incidents_per_year.sort_values(by='Year').reset_index(drop=True)

    time_series = px.line(incidents_per_year, x='Year', y='incidents', title='Terrorist Incidents per Year')
    time_series.update_layout(autosize=True)
    make_transparent(time_series)

    return time_series

@app.callback(
    Output('bar-chart-explore', 'figure'),
    [Input("decade", "value"),
    Input("regions-dropdown", "value"),],
    
)
def display_bar_chart_explore(years, regions):
    df = terror_df[terror_df['Year'].isin(years)]
    if len(regions) > 0:
        df = df[df['Region'].isin(regions)]
    bar_chart = px.bar(df, x="Region", y="casualties", color='AttackType',
             hover_data=["Wounded",'Killed', "Country", 'city'],
             title='Numbers of Casualties by Attack Type')
    make_transparent(bar_chart)

    return bar_chart

@app.callback(
    Output('pie-chart', 'figure'),
    [Input("decade", "value"),
    Input("regions-dropdown", "value"),],
    
)
def display_pie_chart_explore(years, regions):
    df = terror_df[terror_df['Year'].isin(years)]
    if len(regions) > 0:
        df = df[df['Region'].isin(regions)]
    pie_chart = px.pie(df[df['casualties']>50], values='casualties', names='Target_type', 
        title='Proportion of Casualties by Target Type')
    make_transparent(pie_chart)

    return pie_chart

@app.callback(
    Output("cluster-graph", "figure"),
    [Input("cluster-count", "value"),],
)
def make_graph(n_clusters):
    test = pd.DataFrame(terror_df.Group.value_counts()).iloc[1:48,:]
    test.reset_index(level=0, inplace=True)
    z_n_groups = stats.zscore(test.Group.values)
    z_casualities_per_year = stats.zscore(incidents_per_year['casualties'].values)
    clustering_data = pd.DataFrame(z_casualities_per_year.T, columns=['Casualties in a single year'])
    clustering_data['Most groups in a single year'] = z_n_groups

    # minimal input validation, make sure there's at least one cluster
    x = 'Casualties in a single year'
    y = 'Most groups in a single year'
    km = KMeans(n_clusters=max(n_clusters, 1))
    df = clustering_data.loc[:, [x, y]]
    km.fit(df.values)
    df["cluster"] = km.labels_

    centers = km.cluster_centers_

    data = [
        go.Scatter(
            x=df.loc[df.cluster == c, x],
            y=df.loc[df.cluster == c, y],
            mode="markers",
            marker={"size": 8},
            name="Cluster {}".format(c+1),
        )
        for c in range(n_clusters)
    ]

    data.append(
        go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode="markers",
            marker={"color": "#000", "size": 12, "symbol": "diamond"},
            name="Cluster centers",
        )
    )

    layout = {
        "xaxis": {"title": x}, 
        "yaxis": {"title": y}, 
        'paper_bgcolor': 'rgba(0, 0, 0, 0)', 
        "title": 'K-means Clustering',
        "font_color": "#FAF9F6",
        "height": 500}

    return go.Figure(data=data, layout=layout)

@app.callback(
    Output('text-bar','figure'),
    Output('text-bar-graph','style'),
    [Input('cytoscape','tapNodeData'),
    Input("decade", "value"),
    Input("regions-dropdown", "value"),]
)
def update_nodes(data, years, regions):
    df = terror_df[terror_df['Year'].isin(years)]
    if len(regions) > 0:
        df = df[df['Region'].isin(regions)]
    incident_summaries = df['Summary'].dropna()
    corpus = []
    summary = incident_summaries.str.split()
    summary = [word.lower() for i in summary for word in i]
    corpus = summary

    dic = defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word] +=1

    counter = Counter(corpus)
    most = counter.most_common()

    x_word, y_word = [], []
    for word, count in most[:40]:
        if(word not in stop):
            x_word.append(word)
            y_word.append(count)

    x_word = np.array(x_word).T
    y_word = np.array(y_word).T

    top_words = pd.DataFrame(y_word, index=x_word, columns=['Frequency'])
    word_freq = top_words['Frequency'].values
    top_words['Frequency'] = stats.zscore(top_words['Frequency'].values)

    if data is None:
        return dash.no_update
    else:
        dff = top_words.copy()
        dff['Count'] = word_freq
        dff.loc[dff.index == data['label'], 'color'] = "blue"
        fig = px.bar(dff, x='Count', y=dff.index, title='40 Most Frequent Words from Incident Summary', height=550)
        fig.update_traces(marker={'color': dff['color']})
        make_transparent(fig)
        return fig, {"display":"block"}

@app.callback(
    Output("3d-scatter", "figure"),
    [Input("cluster-count-gmm", "value"),],
)
def make_gmm_graph(n_components):
    num_incidents  = pd.DataFrame(terror_df['Year'].value_counts()).reset_index().sort_values('index')
    killed_per_year= terror_df[['Year','Killed']].groupby(['Year'],axis=0).sum().reset_index()
    wounded_per_year= terror_df[['Year','Wounded']].groupby(['Year'],axis=0).sum().reset_index()

    z_num_incidents = stats.zscore(num_incidents['Year'].values)
    z_wounded_per_year = stats.zscore(wounded_per_year['Wounded'].values)
    z_killed_per_year = stats.zscore(killed_per_year['Killed'].values)

    clustering_3d = pd.DataFrame(z_num_incidents.T, columns=['Incidents in a single year'])
    clustering_3d['Wounded in a single year']= z_wounded_per_year
    clustering_3d['Killed in a single year'] =  z_killed_per_year
    X = clustering_3d[['Incidents in a single year', 'Wounded in a single year', 'Killed in a single year']]

    # Set the model and its parameters - 4 clusters
    model4 = GaussianMixture(n_components=n_components, # this is the number of clusters
                            covariance_type='full', # {‘full’, ‘tied’, ‘diag’, ‘spherical’}, default=’full’
                            max_iter=100, # the number of EM iterations to perform. default=100
                            n_init=1, # the number of initializations to perform. default = 1
                            init_params='kmeans', # the method used to initialize the weights, the means and the precisions. {'random' or default='k-means'}
                            verbose=0, # default 0, {0,1,2}
                            random_state=1 # for reproducibility
                            )

    # Fit the model and predict labels
    clust4 = model4.fit(X)
    labels4 = model4.predict(X)
    # smpl=model4.sample(n_samples=20)
    clustering_3d['Clust']=labels4

    scatter_3d = px.scatter_3d(clustering_3d, x='Incidents in a single year', y='Wounded in a single year',
        z='Killed in a single year',title='3D GMM Clustering',
        color = clustering_3d['Clust'], height=500)
    make_transparent(scatter_3d)

    return scatter_3d


@app.callback(
    Output("cytoscape", "elements"),
    [Input("decade", "value"),
    Input("regions-dropdown", "value"),]
)
def display_network(years, regions):
    df = terror_df[terror_df['Year'].isin(years)]
    if len(regions) > 0:
        df = df[df['Region'].isin(regions)]
    incident_summaries = df['Summary'].dropna()
    corpus = []
    summary = incident_summaries.str.split()
    summary = [word.lower() for i in summary for word in i]
    corpus = summary

    dic = defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word] +=1

    counter = Counter(corpus)
    most = counter.most_common()

    x_word, y_word = [], []
    for word, count in most[:40]:
        if(word not in stop):
            x_word.append(word)
            y_word.append(count)

    x_word = np.array(x_word).T
    y_word = np.array(y_word).T

    top_words = pd.DataFrame(y_word, index=x_word, columns=['Frequency'])
    top_words['Frequency'] = stats.zscore(top_words['Frequency'].values)

    pairwise = pd.DataFrame(
        squareform(pdist(top_words)),
        columns = top_words.index,
        index = top_words.index
    )
    # move to long form
    long_form = pairwise.unstack()

    # rename columns and turn into a dataframe
    long_form.index.rename(['Word1', 'Word2'], inplace=True)
    long_form = long_form.to_frame('distance').reset_index()

    long_form = long_form[
        (long_form['distance'] < 0.05) 
        & (long_form['Word1'] != long_form['Word2'])
    ].reset_index(drop=True)

    nodes = [
        {
            'data': {'id': word, 'label': word, 'classes': word},
        }
        for word in long_form.Word1.unique()
    ]

    edges = [
        {'data': {'source': row['Word1'], 'target': row['Word2'], 'weight': '{0:.3f}'.format(row['distance']), 
                'classes':str(index)}}
        for index, row in long_form.iterrows()
    ]

    elements = nodes + edges
    
    return elements

@app.callback(
    Output("hist-graph", "figure"),
    [Input("decade", "value"),
    Input("regions-dropdown", "value"),]
)
def display_hist_graph(years, regions):
    df = terror_df[terror_df['Year'].isin(years)]
    if len(regions) > 0:
        df = df[df['Region'].isin(regions)]
    df = df[df['casualties'].between(5,100)]
    z_casualties = stats.zscore(df['casualties'].values)
    df['casualties'] = z_casualties
    hist_graph = px.histogram(df, x='casualties', marginal='violin', hover_data=['Date'])
    make_transparent(hist_graph)
    return hist_graph

@app.callback(
    Output("boxplot", "figure"),
    [Input("decade", "value"),
    Input("regions-dropdown", "value"),]
)
def display_boxplot(years, regions):
    df = terror_df[terror_df['Year'].isin(years)]
    if len(regions) > 0:
        df = df[df['Region'].isin(regions)]
    z_casualties = stats.zscore(df['casualties'].values)
    df['casualties'] = z_casualties
    boxplot = px.box(df, x='PropertyDamageExtent', y='casualties', title='Distribution of Casualties According to Damage')
    make_transparent(boxplot)
    return boxplot

@app.callback(
    Output("heatmap", "figure"),
    [Input("decade", "value"),
    Input("regions-dropdown", "value"),]
)
def display_heatmap(years, regions):
    df = terror_df[terror_df['Year'].isin(years)]
    if len(regions) > 0:
        df = df[df['Region'].isin(regions)]
    heatmap = px.imshow(df.corr(), title="Correlation Matrix")
    make_transparent(heatmap)
    return heatmap

@app.callback(
    Output("regression", "figure"),
    [Input("regions-dropdown", "value"),]
)
def display_regression_graph(regions):
    df = terror_df[terror_df['Region'].isin(regions)]
    regression_graph = px.scatter(incidents_per_year, x='Year', y='casualties',marginal_y='violin',
                title='Regression Analysis', trendline='ols')
    make_transparent(regression_graph)
    return regression_graph

@app.callback(
    Output('contingency-table', 'figure'),
    Output('chisquare-card', 'children'),
    [Input('ctrl-left', 'value'),
    Input('ctrl-right', 'value'),
    Input("decade", "value"),
    Input("regions-dropdown", "value"),]
)
def display_contingency_table(cat1, cat2, years, regions):
    df = terror_df[terror_df['Year'].isin(years)]
    if len(regions) > 0:
        df = df[df['Region'].isin(regions)]
    c1 = df[cat1].values
    c2 = df[cat2].values

    contingency_table = pd.crosstab(c1, c2)
    fig = ff.create_annotated_heatmap(contingency_table.values,x=contingency_table.columns.values.tolist(), 
                                  y=contingency_table.index.values.tolist(),colorscale='YlOrBr')
    fig.update_layout({
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'font_color': '#FAF9F6',
        'title_text': 'Contingency Table',
    },)

    stat, p, dof, expected = stats.chi2_contingency(contingency_table)
    prob = 0.95
    critical = stats.chi2.ppf(prob, dof)
    message = ''
    # interpret p-value
    alpha = 1.0 - prob
    if p <= alpha:
        message = 'Dependent (reject H0)'
    else:
        message = 'Independent (fail to reject H0)'
    results_card = dbc.Card(
        [
            dbc.CardHeader("Chi-Square Test Results", 
                style={"margin-top": "0","text-align": "center", "font-weight": "bold", "color": "#FAF9F6"}
            ),
            dbc.CardBody(
                [
                    html.P("Degrees of Freedom = {}".format(dof), className="card-text"),
                    html.P("Alpha = {0:.3f}".format(alpha), className="card-text"),
                    html.P("Critical Value = {0:.3f}".format(critical), className="card-text"),
                    html.P("Chi-Square Statistic = {0}".format(stat), className="card-text"),
                ]
            , style={"color": "#FAF9F6"}),
            dbc.CardFooter("Conclusion: {}".format(message), style={"font-weight": "bold", "color": "#FAF9F6"}),
        ],
        style={"width": "18rem"},
    )
    return fig, results_card

@app.callback(
    Output('decision-tree', 'figure'),
    Output('classifier-graph', 'style'),
    Output('tree-results', 'children'),
    [
        Input("train-classifier", "n_clicks"),
        Input("decade", "value"),
        Input("regions-dropdown", "value"),
    ],
    State('ctrl-classifier', 'value'),
)
def train_and_display(n, years, regions, features):
    df = terror_df[terror_df['Year'].isin(years)].dropna()
    if len(regions) > 0:
        df = df[df['Region'].isin(regions)]
    if n:
        X = df[features]
        y = df['success']

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

        model = DecisionTreeClassifier(random_state=0)
        model.fit(X_train, y_train)

        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)

        fig = px.bar(x=X_train.columns.to_list(), y=model.feature_importances_,
            labels=dict(x='Feature', y='Feature Importance'),
            title='Weight of each feature for predicting incident success'
        )
        make_transparent(fig)

        cards = dbc.Container(children=[
                        dbc.Row(dbc.Card(create_card('bi bi-graph-up', '   Training Accuracy', '{0:.4f}'.format(train_accuracy)), color="info", inverse=True)),
                        html.Br(),
                        dbc.Row(dbc.Card(create_card('bi bi-clipboard-check', '   Testing Accuracy', '{0:.4f}'.format(test_accuracy)), color="warning", inverse=True)),
                        html.Br(),
                        dbc.Row(dbc.Card(create_card('bi bi-funnel', '   Train/Test Split', '70/30'), color="success", inverse=True)),
                ], fluid=True)
            

        return fig, {"display":"block"}, cards
    else:
        return go.Figure(), {"display":"none"}, dbc.Container()

@app.callback(
    Output("bayes", "figure"),
    [Input("decade", "value"),
    Input("regions-dropdown", "value"),
    Input("lamda-prior", "value"),
    Input("lamda", "value"),]
)
def display_bayes_graphs(years, regions, prior_rate, rate):
    df = terror_df[terror_df['Year'].isin(years)].dropna()
    if len(regions) > 0:
        df = df[df['Region'].isin(regions)]
    most_freq_casualties = df[df['casualties'].between(1, 5)]
    x_cas = most_freq_casualties['casualties'].values
    x_cas = x_cas[:20]

    # likelihood
    x_likelihood = np.arange(0,7, 1)
    likelihood = stats.poisson.pmf(x_likelihood, rate)

    #Prior
    x_prior = np.arange(0,5, 1)
    probability = stats.poisson.pmf(x_likelihood, prior_rate)

    rate_candidates = np.arange(0,rate, 1)

    def posterior(rate, x):
        mu_prior = stats.poisson.pmf(x, prior_rate,)
        return mu_prior * stats.poisson.pmf(x, rate)
        
    def compute_posteriors(candidates, x):
        for rate in candidates:
            yield posterior(rate, x)

    posteriors = list(compute_posteriors(rate_candidates, x_cas))
    posteriors = [np.unique(x) for x in posteriors]
    posteriors = np.concatenate(posteriors).ravel()

    bayes = go.Figure()

    # Add traces
    bayes.add_trace(go.Scatter(x=x_likelihood, y=likelihood,
                        mode='lines+markers',
                        name='Likelihoods'))
    bayes.add_trace(go.Scatter(x=x_prior, y=probability,
                        mode='lines+markers',
                        name='Informed prior'))
    bayes.add_trace(go.Scatter(x=rate_candidates, y=posteriors,
                        mode='lines+markers',
                        name='Posterior distribution'))
    bayes.update_layout(title="Poisson Distributions", xaxis_title="Casualties", yaxis_title="Probabilities", legend_title="Distributions"
                        , paper_bgcolor='rgba(0, 0, 0, 0)', font_color='#FAF9F6')
    return bayes
# # make sure that x and y values can't be the same variable
# def filter_options(v):
#     """Disable option v"""
#     return [
#         {"label": col, "value": col, "disabled": col == v}
#         for col in clustering_data.columns
#     ]


# # functionality is the same for both dropdowns, so we reuse filter_options
# app.callback(Output("x-variable", "options"), [Input("y-variable", "value")])(
#     filter_options
# )
# app.callback(Output("y-variable", "options"), [Input("x-variable", "value")])(
#     filter_options
# )

@app.callback(
    [
        Output("sidebar", "style"),
        Output("page-content", "style"),
        Output("side_click", "data"),
    ],

    [Input("btn_sidebar", "n_clicks")],
    [State("side_click", "data"),]
)
def toggle_sidebar(n, nclick):
    if n:
        if nclick == "SHOW":
            sidebar_style = SIDEBAR_HIDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"
    else:
        sidebar_style = SIDEBAR_HIDEN
        content_style = CONTENT_STYLE1
        cur_nclick = "HIDDEN"

    return sidebar_style, content_style, cur_nclick

# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
# @app.callback(
#     [Output(f"page-{i}-link", "active") for i in range(1, 4)],
#     [Input("url", "pathname")],
# )
# def toggle_active_links(pathname):
#     if pathname == "/":
#         # Treat page 1 as the homepage / index
#         return True, False, False
#     return [pathname == f"/page-{i}" for i in range(1, 4)]

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return [row_explore]
    elif pathname == "/analyze":
        return [row_analyze]
    elif pathname == "/predict":
        return [row_predict]
    elif pathname == "/video":
        return [row_video]
    # If the user tries to reach a different page, return a 404 message
    return dbc.Container(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} is pretty sus my friend...", className='lead'),
        ]
    )

# Running the server
if __name__ == "__main__":
    app.run_server(debug=True, port=8060)