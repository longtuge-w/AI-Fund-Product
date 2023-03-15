import pandas as pd
import numpy as np
import pickle
import time
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import dash_table
import quantstats as qs
from VisualInfo import VisualInfo


class visualize(VisualInfo):
    def __init__(self,time_data,Stock_dict,FacRet_dict,FacVar_dict,IndustryWeight_dict,Info_df):
        VisualInfo.__init__(self)
        self.current = self.start_date
        self.last_date = self.start_date
        self.time_data = time_data
        self.Stock_dict = Stock_dict
        self.FacRet_dict = FacRet_dict
        self.FacVar_dict = FacVar_dict
        self.IndustryWeight_dict = IndustryWeight_dict
        self.Info_df = Info_df
        self.app = dash.Dash(__name__, external_stylesheets=self.external_stylesheets)
        self.portfolio_dict = {'Portfolio_1':'1','Portfolio_2':'2','Portfolio_3':'3','Portfolio_4':'4'
                ,'Portfolio_5':'5','Portfolio_6':'6','Portfolio_7':'7'
                ,'Portfolio_8':'8','Portfolio_9':'9','Portfolio_10':'10'}


    def set_startDate(self,startDate):
        self.start_date = startDate


    def set_endDate(self,endDate):
        self.end_date = endDate


    def set_window(self,window):
        self.window = window

    
    def set_format(self,dataFormat):
        self.format = dataFormat


    def set_numPortfolio(self,numPortfolio):
        self.NumPortfolio = numPortfolio


    # turn the format of one column into Timestamp
    def get_date_df(self,df,column='date'):
        df[column] = df[column].apply(lambda x: pd.Timestamp(int(str(x)[:4]),int(str(x)[4:6]),int(str(x)[6:])))
        return df


    # get the excess return within a specific period
    def get_excess_return(self,start,end,p_lst):
        money = [0]*len(p_lst)
        start_index = self.time_data.index(start)
        end_index = self.time_data.index(end)

        # the original amount of money
        origin_money = [(self.Stock_dict[start][int(self.portfolio_dict[i])-1]['w_%s'%(self.portfolio_dict[i])]\
            *self.Stock_dict[start][int(self.portfolio_dict[i])-1]['pclose']).sum() for i in p_lst]
            
        # calculate the return based on the amount of money
        while start_index<end_index:
            if start_index + self.window <= end_index:
                temp_end_index = start_index + self.window
            else:
                temp_end_index = end_index
            start_lst = self.Stock_dict[self.time_data[start_index]]
            end_lst = self.Stock_dict[self.time_data[temp_end_index]]
            for i in range(len(p_lst)):
                index_ = int(self.portfolio_dict[p_lst[i]])-1
                # value of stocks on last window period
                start_money = (start_lst[index_]['w_%s'%(index_+1)]*start_lst[index_]['pclose']).sum()
                # value of stocks on current window period
                end_money = (start_lst[index_]['w_%s'%(index_+1)]*end_lst[index_]['pclose']).sum()
                money[i] = money[i]-start_money+end_money
            start_index += self.window
        # calculate the benchmark return within this period
        bench_ret = (self.Info_df.loc[end,'bench']-self.Info_df.loc[start,'bench'])/self.Info_df.loc[start,'bench']
        return [money[i]/origin_money[i]-bench_ret for i in range(len(p_lst))]


    def get_visual_opt(self):
        self.app.layout = html.Div([

            # input a trading date, visualize the performance on this date
            html.P(u'Enter date with format "yyyymmdd": '),
            dcc.Input(id='date', value=self.current, type='number', debounce=True, min=1, step=1),
            html.Button(id='submit_1', n_clicks=0, children='Submit'),
            html.P(id='err', style={'color': 'red'}),
            html.P(id='out'),

            # input start trading date and end trading date, visualize the performance within this period
            html.P(u'Enter start date and end date with format "yyyymmdd" to check the performance of 3 portfolios: '),
            dcc.Input(id='start', value=self.start_date, type='number', debounce=True, min=1, step=1),
            dcc.Input(id='end', value=self.end_date, type='number', debounce=True, min=1, step=1),
            html.Button(id='submit_2', n_clicks=0, children='Submit'),
            html.P(id='err_2', style={'color': 'red'}),
            html.P(id='out_2'),

            # user can choose portfolio they want by the dropdown
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='p1',
                        options=[{'label': 'Portfolio_%s'%(i+1), 'value': 'Portfolio_%s'%(i+1)} for i in range(self.NumPortfolio)]\
                            +[{'label': 'None', 'value': 'None'}],
                        value='Portfolio_1'
                    )],
                style={'width': '24%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Dropdown(
                        id='p2',
                        options=[{'label': 'Portfolio_%s'%(i+1), 'value': 'Portfolio_%s'%(i+1)} for i in range(self.NumPortfolio)]\
                            +[{'label': 'None', 'value': 'None'}],
                        value='None'
                    )],
                style={'width': '24%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Dropdown(
                        id='p3',
                        options=[{'label': 'Portfolio_%s'%(i+1), 'value': 'Portfolio_%s'%(i+1)} for i in range(self.NumPortfolio)]\
                            +[{'label': 'None', 'value': 'None'}],
                        value='None'
                    )],
                style={'width': '24%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Dropdown(
                        id='p4',
                        options=[{'label': 'Portfolio_%s'%(i+1), 'value': 'Portfolio_%s'%(i+1)} for i in range(self.NumPortfolio)]\
                            +[{'label': 'None', 'value': 'None'}],
                        value='None'
                    )],
                style={'width': '24%', 'display': 'inline-block'})
            ]),

            # bar chart of the factor return
            html.Div([
                html.Div(dcc.Graph(id='bar_chart_ind')),
            ]),
            html.Div([
                html.Div(dcc.Graph(id='bar_chart_style')),
            ]),

            # pie chart of the industry
            html.Div([
                html.Div(dcc.Graph(id='pie_chart_ind_1'),className="six columns"),
                html.Div(dcc.Graph(id='pie_chart_ind_2'),className="six columns"),
                html.Div(dcc.Graph(id='pie_chart_ind_3'),className="six columns"),
                html.Div(dcc.Graph(id='pie_chart_ind_4'),className="six columns")
            ],className='row'),

            # line chart of the variance of factor return
            html.Div([
                html.Div(dcc.Graph(id='line_chart_ind')),
            ]),
            html.Div([
                html.Div(dcc.Graph(id='line_chart_style')),
            ]),

            # summary table of the performance of portfolio
            html.Div([
                dash_table.DataTable(
                    id='table',
                    columns=[{
                        'name': '{}'.format(self.table_columns[i-1]),
                        'id': 'column-{}'.format(i),
                        'deletable': True,
                        'renamable': True
                    } for i in range(1, 8)],
                    editable=True
                    )
                ]),
                
            # line chart of the return of portfolio
            html.Div([
                html.Div(dcc.Graph(id='line_chart_summary')),
            ]),
            dcc.Store(id='signal')
        ])


        # first submit button
        @self.app.callback(
            Output('out', 'children'),
            Output('err', 'children'),
            Input('submit_1', 'n_clicks'),
            State('date', 'value')
        )
        def show_factors(n_click,date):
            if date is None:
                raise dash.exceptions.PreventUpdate

            if date not in self.FacRet_dict.keys():
                return '', 'Data from "{}" is not founded! Please try another trading day and make sure your input satisfies the format "yyyymmdd".'.format(date)
            return 'Data from "{}" is updated!'.format(date), ''


        # second submit button
        @self.app.callback(
            Output('out_2', 'children'),
            Output('err_2', 'children'),
            Input('submit_2', 'n_clicks'),
            State('start', 'value'),
            State('end', 'value')
        )
        def show_factors(n_click,start,end):
            if start is None or end is None:
                raise dash.exceptions.PreventUpdate

            if start not in self.FacRet_dict.keys() or end not in self.FacRet_dict.keys() or start >= end:
                return '', 'Data from "{}" to "{}" is not founded! Please try another trading day and make sure your input satisfies the format "yyyymmdd".'.format(start,end)
            return 'Data from "{}" to "{}" is updated!'.format(start,end), ''


        # bar chart of the factor return
        @self.app.callback(Output('bar_chart_ind', 'figure'),
                    Output('bar_chart_style', 'figure'),
                    Input('submit_1', 'n_clicks'),
                    Input('p1','value'),
                    Input('p2','value'),
                    Input('p3','value'),
                    Input('p4','value'),
                    State('date', 'value'))
        def update_graph_1(n_clicks, p1, p2, p3, p4, input):
            data = self.FacRet_dict[int(input)]
            industry = data.iloc[:len(self.Industry)].copy()
            style = data.iloc[len(self.Industry):].copy()
            p_lst = [i for i in [p1,p2,p3,p4] if i != 'None']

            fig1 = px.bar(industry, x="factor", y=['ret_%s'%(self.portfolio_dict[i]) for i in p_lst], barmode="group")
            fig2 = px.bar(style, x="factor", y=['ret_%s'%(self.portfolio_dict[i]) for i in p_lst], barmode="group")

            return fig1, fig2


        # pie chart of the industry
        @self.app.callback(Output('pie_chart_ind_1', 'figure'),
                    Output('pie_chart_ind_2', 'figure'),
                    Output('pie_chart_ind_3', 'figure'),
                    Output('pie_chart_ind_4', 'figure'),
                    Input('submit_1', 'n_clicks'),
                    Input('p1','value'),
                    Input('p2','value'),
                    Input('p3','value'),
                    Input('p4','value'),
                    State('date', 'value'))
        def update_graph_2(n_clicks, p1, p2, p3, p4, input):
            data = self.IndustryWeight_dict[int(input)]

            if p1 != 'None':
                fig1 = px.pie(data, values='w_%s'%(self.portfolio_dict[p1]), names='factor',\
                    title='The return of the portfolio %s'%(self.portfolio_dict[p1]),hole=.7)
            else:
                fig1 = px.pie()

            if p2 != 'None':
                fig2 = px.pie(data, values='w_%s'%(self.portfolio_dict[p2]), names='factor',\
                    title='The return of the portfolio %s'%(self.portfolio_dict[p2]),hole=.7)
            else:
                fig2 = px.pie()

            if p3 != 'None':
                fig3 = px.pie(data, values='w_%s'%(self.portfolio_dict[p3]), names='factor',\
                    title='The return of the portfolio %s'%(self.portfolio_dict[p3]),hole=.7)
            else:
                fig3 = px.pie()

            if p4 != 'None':
                fig4 = px.pie(data, values='w_%s'%(self.portfolio_dict[p4]), names='factor',\
                    title='The return of the portfolio %s'%(self.portfolio_dict[p4]),hole=.7)
            else:
                fig4 = px.pie()

            return fig1, fig2, fig3, fig4


        # line chart of the variance of factor return
        @self.app.callback(Output('line_chart_ind', 'figure'),
                    Output('line_chart_style', 'figure'),
                    Input('submit_2', 'n_clicks'),
                    Input('p1','value'),
                    Input('p2','value'),
                    Input('p3','value'),
                    Input('p4','value'),
                    State('start', 'value'),
                    State('end', 'value'))
        def update_graph_3(n_clicks, p1, p2, p3, p4, start, end):
            data = pd.DataFrame()
            start_index = self.time_data.index(start)
            end_index = self.time_data.index(end)+1
            time_lst = self.time_data[start_index:end_index]
            for date in time_lst:
                data = data.append(self.FacVar_dict[date])
            data = self.get_date_df(data)
            p_lst = [i for i in [p1,p2,p3,p4] if i != 'None']

            fig1 = px.line(data, x='date', y=['var_%s_ind'%(self.portfolio_dict[i]) for i in p_lst], \
                title='The variance of the portfolios (Industry)')
            fig2 = px.line(data, x='date', y=['var_%s_sty'%(self.portfolio_dict[i]) for i in p_lst], \
                title='The variance of the portfolios (Style)')

            return fig1, fig2


        # line chart of the return of portfolio
        @self.app.callback(Output('line_chart_summary', 'figure'),
                    Input('submit_2', 'n_clicks'),
                    Input('p1','value'),
                    Input('p2','value'),
                    Input('p3','value'),
                    Input('p4','value'),
                    State('start', 'value'),
                    State('end', 'value'))
        def update_graph_4(n_clicks, p1, p2, p3, p4, start, end):
            data = self.Info_df.loc[start:end].copy()
            p_lst = [i for i in [p1,p2,p3,p4] if i != 'None']
            data = self.get_date_df(data)

            fig = px.line(data, x='date', y=['ret_%s'%(self.portfolio_dict[i]) for i in p_lst], \
                title='The performance of the portfolios')

            return fig


        # summary table of the performance of portfolio
        @self.app.callback(Output('table', 'data'),
                    Input('submit_2', 'n_clicks'),
                    Input('p1','value'),
                    Input('p2','value'),
                    Input('p3','value'),
                    Input('p4','value'),
                    State('start', 'value'),
                    State('end', 'value'))
        def update_table(n_clicks, p1, p2, p3, p4, start, end):
            data = self.Info_df.loc[start:end].copy()
            data = self.get_date_df(data)
            end_Info_df  = self.Info_df.loc[end]
            data.reset_index(inplace=True)
            data.set_index('date',inplace=True)
            table_lst, p_lst = [], [i for i in [p1,p2,p3,p4] if i != 'None']
            money_lst = self.get_excess_return(start,end,p_lst)

            for i in range(len(p_lst)):
                table_lst.append({'column-1':'Portfolio_%s'%(self.portfolio_dict[p_lst[i]]),
                'column-2':str(round(money_lst[i]*100,2))+'%',
                'column-3':round(end_Info_df['PE_%s'%(self.portfolio_dict[p_lst[i]])],2),
                'column-4':round(end_Info_df['PB_%s'%(self.portfolio_dict[p_lst[i]])],2),
                'column-5':round(end_Info_df['ROE_%s'%(self.portfolio_dict[p_lst[i]])],2),
                'column-6':round(qs.stats.information_ratio(data['ret_%s'%(self.portfolio_dict[p_lst[i]])],data['bench_ret']),2),
                'column-7':round(qs.stats.calmar(data['ret_%s'%(self.portfolio_dict[p_lst[i]])]),2)})

            return table_lst


    def main(self):
        self.get_visual_opt()
        self.app.run_server(debug=False)


# if __name__ == '__main__':
#     visual = visualize()
#     visual.main()