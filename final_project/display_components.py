from dash import dcc, html
import dash_bootstrap_components as dbc

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 62.5,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": "#073642",
}

SIDEBAR_HIDEN = {
    "position": "fixed",
    "top": 62.5,
    "left": "-16rem",
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0rem 0rem",
    "background-color": "#073642",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "transition": "margin-left .5s",
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": 'transparent',
}

CONTENT_STYLE1 = {
    "transition": "margin-left .5s",
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "transparent",
}

# Defining navbar items
explore_button = dbc.NavItem(dbc.NavLink('Explore',href="/", active='exact'))
analyze_button = dbc.NavItem(dbc.NavLink('Analyze',href="/analyze", active='exact'))
predict_button = dbc.NavItem(dbc.NavLink('Predict',href="/predict", active='exact'))

# Functions
def create_menu(explore_button, analyze_button, predict_button):
    menu_items = dbc.Row(
        [
            dbc.Col(explore_button),
            dbc.Col(analyze_button),
            dbc.Col(predict_button),
            dbc.Col(
                dbc.Button(
                    html.I(className='bi bi-list'),color="primary", className="ms-2", n_clicks=0, id="btn_sidebar"
                ),
                width="auto",
            ),
        ],
        className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
        align="center",
    )
    return menu_items

def create_navbar():
    navbar = dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    # Use row and col to control vertical alignment of logo / brand
                    dbc.Row(
                        [
                            dbc.Col(html.I(className='bi bi-globe')),
                            dbc.Col(dbc.NavbarBrand("Global Terrorism Dashboard", className="ms-2", id='nav-title')),
                        ],
                        align="center",
                        className="g-0"
                    ),
                    href="https://www.start.umd.edu/gtd/",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                dbc.Tooltip(
                    "Click "
                    "to go to the Global Terrorism Database's official webpage.",
                    target="nav-title",
                ),
                dbc.Collapse(
                    create_menu(
                        explore_button,
                        analyze_button,
                        predict_button
                    ),
                    id="navbar-collapse",
                    is_open=False,
                    navbar=True,
                ),
            ],
        ), 
        color="dark",
        dark=True,
        className='navbar-change',
        expand='lg',
        sticky='top'
    )
    return navbar

#f8f9fa --> white
def create_sidebar_controls(df):
    controls = dbc.Card(
        [
            html.H3('Controls'),
            html.Div(
                [
                    dbc.Label("X variable"),
                    dcc.Dropdown(
                        id="x-variable",
                        options=[
                            {"label": col, "value": col} for col in df.columns
                        ],
                        value=df.columns[0],
                    ),
                ]
            ),
            html.Div(
                [
                    dbc.Label("Y variable"),
                    dcc.Dropdown(
                        id="y-variable",
                        options=[
                            {"label": col, "value": col} for col in df.columns
                        ],
                        value=df.columns[1],
                    ),
                ]
            ),
            html.Div(
                [
                    dbc.Label("Cluster count"),
                    dbc.Input(id="cluster-count", type="number", value=3),
                ]
            ),
        ],
        body=True,
    )
    return controls

# Card
def create_card(icon, header, title):
    card_content = [
        dbc.CardHeader(children=[ html.I(className=icon), header], style={"font-weight": "bold",'text-align':'center'}),
        dbc.CardBody(
            [
                html.H4(title, className="card-title", style={'textAlign':'center'}),
            ]
        ),
    ]
    return card_content

    
    # options=[{'value': num, 'label': '{}s'.format(num)} for num in df['Region'].unique().tolist()],
    # placeholder="Select a country",
    # value='MTL',
    # multi=True