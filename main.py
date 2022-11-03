import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("covid_19_clean_complete.csv", parse_dates=['Date'])
df.head()
#função para remover caracteres especiais e deixar letras minusculas
def corrige_colunas(col_name):
  return re.sub(r"[/| ]", "", col_name).lower()

df.columns = [corrige_colunas(col) for col in df.columns]

df.loc[df.countryregion == "Brazil"]
brasil = df.loc[(df.countryregion == "Brazil") & (df.confirmed > 0)]

px.line(brasil, 'date', 'deaths', title='casos confirmados no Brasil')

brasil['novasmortes'] = list(map(
    lambda x: 0 if (x==0) else brasil['deaths'].iloc[x] - brasil['deaths'].iloc[x-1],
    np.arange(brasil.shape[0])
))

px.line(brasil,'date','novasmortes', title='Novas mortes por dia')

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=brasil.date, y=brasil.deaths, name='Mortes', mode='lines+markers', line={'color':'red'})
)

fig.update_layout(title='Mortes por COVID-19 no Brasil')
fig.show()


def taxa_cresimento(data, variable, data_inicio=None, data_fim=None):
  if data_inicio == None:
    data_inicio = data.date.loc[data[variable] > 0].min()
  else:
    data_inicio = pd.to_datetime(data_inicio)

  if data_fim == None:
    data_fim = data.date.iloc[-1]
  else:
    data_fim = pd.to_datetime(data_fim)

  passado = data.loc[data.date == data_inicio, variable].values[0]
  presente = data.loc[data.date == data_fim, variable].values[0]

  n = (data_fim - data_inicio).days

  taxa = (presente/passado)**(1/n) -1

  return taxa * 100

taxa_cresimento(brasil, 'deaths')


def taxa_cresimento_diario(data, variable, data_inicio=None):
  if data_inicio == None:
    data_inicio = data.date.loc[data[variable] > 0].min()
  else:
    data_inicio = pd.to_datetime(data_inicio)

  data_fim = data.date.max()
  n = (data_fim - data_inicio).days
  taxas = list(map(
    lambda x: (data[variable].iloc[x] - data[variable].iloc[x - 1]) / data[variable].iloc[x - 1],
    range(1, n + 1)
  ))

  return np.array(taxas) * 100

tx_dia = taxa_cresimento_diario(brasil, 'confirmed')

primeiro_dia = brasil.date.loc[brasil.confirmed > 0].min()

px.line(x=pd.date_range(primeiro_dia, brasil.date.max())[1:],
        y=tx_dia, title="taxa de Crescimento dos casos confirmados no Brasil")

confirmados = brasil.confirmed
confirmados.index = brasil.date

res = seasonal_decompose(confirmados)
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(10,8))
ax1.plot(res.observed)
ax2.plot(res.trend)
ax3.plot(res.seasonal)
ax4.plot(confirmados.index, res.resid)
ax4.axhline(0, linestyle='dashed', c='black')
plt.show()


# !pip install pmdarima
from pmdarima.arima import auto_arima

modelo = auto_arima(confirmados)

fig = go.Figure(
    go.Scatter(
        x=confirmados.index, y=confirmados, name='Observados'
    )
)

fig.add_trace(
    go.Scatter(
        x=confirmados.index, y=modelo.predict_in_sample(), name='Preditos'
    )
)

fig.add_trace(
    go.Scatter(
        x=pd.date_range('2020-05-20', '2020-06-20'), y=modelo.predict(31), name='Forcast'
    )
)

fig.update_layout(title='Previsão de casos confirmados no Brasil para 30 dias')
fig.show()