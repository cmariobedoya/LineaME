import pandas as pd
import numpy as np
import seaborn as sns
import holidays
import matplotlib.pyplot as plt
from datetime import timedelta,datetime
import datetime
from pandas.tseries.offsets import MonthEnd, MonthOffset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import ensemble
from sklearn.neural_network import MLPRegressor
import pyodbc
import plotly.plotly as py
import plotly.graph_objs as go
import math

conn = pyodbc.connect('DSN=Impala', autocommit=True)
query = """select * from proceso_serv_para_los_clientes.planilla_linea_me"""
df = pd.read_sql(query, conn)
df.head(5)

def fill_segmento(segmento):
    if pd.isnull(segmento) or (segmento==""):
        return 'otro'
    else:
        return segmento


df['segmento']=df['segmento'].apply(fill_segmento)
df['segmento']=df['segmento'].apply(lambda x: x.strip())
list(df['segmento'].values)

# Rango de fechas para generar el calendario
START_DATE = '2016-01-01'
END_DATE = '2017-12-31'
START_PRED = '2018-01-23'
END_PRED = '2018-01-31'
co_holidays = holidays.CO()

def get_next_holiday(date):
    """
    Devuelve el numero de días que faltan para el siguiente dia
    festivo.

    Ignora festivos en Sabados y Domingos.

    Parametros
    ----------
    date : datetime, fecha que se quiere consultar.

    """
    i = 0
    while date not in co_holidays:
        date += timedelta(days=1)
        i += 1
    return i


def get_last_holiday(date):
    """
    Devuelve el numero de días que pasaron desde el ultimo dia festivo.

    Ignora festivos en Sabados y Domingos.

    Parametros
    ----------
    date : datetime, fecha que se quiere consultar.

    """
    i = 0
    while date not in co_holidays:
        date += timedelta(days=-1)
        i += 1
    return i


def get_especial_holiday(date, holiday):
    """
    Devuelve el numero de días que faltan para el proximo dia especial.

    NO ignora festivos en Sabados y Domingos.

    Parametros
    ----------
    date : datetime, fecha que se quiere consultar.
    holiday : string, Alguno de los siguientes dias: 'Navidad [Christmas]'
                      'Dia Madre', 'Dia Padre'.

    """
    i = 0
    if holiday == 'Navidad [Christmas]':
        while co_holidays.get(date) != holiday:
            date += timedelta(days=1)
            i += 1
        return i
    if holiday == 'Dia Madre':
        while date not in DIAS_MADRE:
            date += timedelta(days=1)
            i += 1
        return i
    if holiday == 'Dia Padre':
        while date not in DIAS_PADRE:
            date += timedelta(days=1)
            i += 1
        return i


def get_quincena(date):
    """
    Cuenta el numero de dias que faltan para el proximo 15 del mes.

    Parametros
    ----------
    date : datetime : fecha que se quiere consultar.

    """
    i = 0
    while date.day != 15:
        date += timedelta(days=1)
        i += 1
    return i

# def replace_hour_am(x):
#     return x.replace('a. m.','am')
#
# def replace_hour_pm(x):
#     return x.replace('p. m.','pm')
#
# def replace_hour_am_m(x):
#     return x.replace('AM','am')
#
# def replace_hour_pm_m(x):
#     return x.replace('PM','pm')

#df['fecha']=df['fecha'].apply(replace_hour_am)
##df['fecha']=df['fecha'].apply(replace_hour_pm)
#df['fecha']=df['fecha'].apply(replace_hour_am_m)
#df['fecha']=df['fecha'].apply(replace_hour_pm_m)
df['FECH_HORA'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y %H:%M')
df[df['FECH_HORA']>'2016-10-01 00:00:00']
df.drop(['codigo_actividad','asesor','duracion','origen','identificacion'],axis=1,inplace=True)
df.loc[df['FECH_HORA'].dt.minute >= 30, 'rango'] = '30'
df.loc[df['FECH_HORA'].dt.minute < 30, 'rango'] = '00'
df['month']=df['FECH_HORA'].dt.month.astype('str')
df['day']=df['FECH_HORA'].dt.day.astype('str')
#CREAMOS EL CAMPO INDICE CON LA FECHA + EL RANGO DE HORA DE LA LLAMADA
df['index'] = (df['FECH_HORA'].dt.year.astype('str') + '-' +
               df['FECH_HORA'].dt.month.astype('str') + '-' +
               df['FECH_HORA'].dt.day.astype('str') + ' ' +
               df['FECH_HORA'].dt.hour.astype('str') + ':' +
               df['rango'])
#df.to_excel("C:/Users/cmbedoya/Documents/basica.xlsx",sheet_name='PREDICCION')
#test=datetime.strptime(df['FECH_HORA'][0], "%d/%m/%Y %H %M")
#AGRUPAMOS POR FECHA -RANGO-SEGMENTO PARA OBTENER EL # DE LLAMADAS DE CADA FRANJA
df_result = df.groupby(by=['index','segmento']).count()[['FECH_HORA']]
#REINICIAMOS EL INDICE
df_result.reset_index(inplace=True)
#CONVERTIMOS LA COLUMNA INDEX A DATETIME PARA PODER EXTRAER INFORMACION POSTERIORMENTE
df_result['index']=(pd.to_datetime(df_result['index']))
#RENOMBRAMOS LAS COLUMNAS
df_result.columns=['fecha','segmento','total_llamadas']
df_result.head(10)
#FILTRAMOS FECHAS DEL 2017-REOMVER INSTRUCCION EN CASO DE QUERER ENTRENAR CON DATOS ANTERIORES
df_result['fecha']=(pd.to_datetime(df_result['fecha']))
df_result=df_result[df_result['fecha']>'2016-12-31 10:00:00']
df_result.head()
#modificamos la fecha desde la que parte el historico, ya que el comportamiento anterior cambió radicalmente y ya no nos porta al momento
#df=df[df['FECH_HORA']>'2017-03-01 00:00:00']
df_result.to_excel("C:/Users/cmbedoya/Documents/reales_basica.xlsx",sheet_name='PREDICCION')

#FUNCION QUE ADICIONA INFORMACION EXTRAIDA DESDE LA FECHA
def variables(df):
    df['day'] = df['fecha'].dt.day
    df['month'] = df['fecha'].dt.month
    df['year'] = df['fecha'].dt.year
    df['dia_semana'] = df['fecha'].dt.weekday_name
    df['festivo'] = df['fecha'].apply(lambda day: day in co_holidays)
    df['n_dias_festivo'] = df['fecha'].apply(lambda date: get_next_holiday(date))
    df['n_dias_festivo_prev'] = df['fecha'].apply(lambda date: get_last_holiday(date))
    df['n_dias_navidad'] = df['fecha'].apply(lambda date: get_especial_holiday(date, 'Navidad [Christmas]'))
    #df['n_dias_dd_madre'] = df['fecha'].apply(lambda date: get_especial_holiday(date, 'Dia Madre'))
    #df['n_dias_dd_padre'] = df['fecha'].apply(lambda date: get_especial_holiday(date, 'Navidad [Christmas]'))
    df['n_dias_quincena'] = df['fecha'].apply(lambda date: get_quincena(date))
    df['n_dias_fin_mes'] = ((df['fecha'] + MonthEnd(1)) - df['fecha']).astype(str).str[0:2].astype(int)
    df['hora'] = df['fecha'].dt.hour
    df['minutos'] = df['fecha'].dt.minute
    #df['quarter'] = df.index.quarter

    return df

#ANEXAMOS VARIABLES AL DF A PARTIR DE LA FECHA
df_result = variables(df_result)
#sns.distplot(df_result['total_llamadas'])
#df_result.set_index(['fecha','segmento'],inplace=True)

#SACAMOS LOS SEGMENTOS UNICOS
lista_segmentos=list(df.segmento.unique())
lista_segmentos

#sns.boxplot(df_result['total_llamadas'],y=df_result.index.get_level_values('segmento'))
#sns.boxplot(df_result['total_llamadas'],y=df_result['segmento'])

#SACAMOS EL QUANTILE 99 PARA ELIMINAR POSTERIORMENTE LOS VALORES MAYORES A ESTE(OUTLIERS)
df_result['total_llamadas'].quantile(0.99)
#sns.distplot(df_result['total_llamadas'])

#ANEXAMOS VARIABLES AL MODELO
df_result['dia_semana'] = df_result['dia_semana'].astype('str')
df_result['day'] = df_result['day'].astype('str')
df_result['month'] = df_result['month'].astype('str')
df_result['year'] = df_result['year'].astype('str')
df_result['hora'] = df_result['hora'].astype('str')


#CONVERTIRMOS LAS VARIABLES CATEGÓRICAS EN DUMMIES PARA PODER ENTRENAR EL MODELO
df_dummies = pd.get_dummies(df_result)
#df_dummies.drop(['fecha'], axis=1,inplace=True)
df_dummies.head(2)

#SEPARAMOS EL DATASET ORIGINAL EN DATOS DE ENTENAMIENTO Y PRUEBA
X_train, X_test, y_train, y_test = train_test_split(df_dummies.drop('total_llamadas', axis=1),
                                                    df_dummies['total_llamadas'], test_size=0.30,
                                                    random_state=101)

#GUARDAMOS LA FECHA EN UNA VARIABLE ANTES DE REMOVERLA
fecha_train=X_train['fecha']
fecha_test=X_test['fecha']
X_train.drop('fecha',axis=1,inplace=True)
X_test.drop('fecha',axis=1,inplace=True)
alpha=0.95
params = {'n_estimators':900, 'max_depth':8, 'min_samples_split': 6,
          'learning_rate': 0.01, 'loss': 'ls'}
"""params = {'n_estimators':800, 'max_depth':10, 'min_samples_split': 6,
          'learning_rate': 0.01, 'loss': 'ls'}"""
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)

predicted_boost=clf.predict(X_test)
r2_score(y_test, predicted_boost)

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

result = pd.DataFrame(y_test)
result['predicho'] = predicted_boost
result.columns=['real','predicho']
result['fecha']=fecha_test
result['nombre_dia'] =result['fecha'].dt.weekday_name
result['fecha_2']=result['fecha'].dt.date
result['day'] = result['fecha'].dt.day
result['month'] = result['fecha'].dt.month
result['year'] = result['fecha'].dt.year
result['hour'] = result['fecha'].dt.hour
result['minute'] = result['fecha'].dt.minute


#SACAMOS LOS SEGMENTOS CORRESPONDIENTES A LA predicción
columnas_segmento=[
       'segmento_1', 'segmento_2', 'segmento_3', 'segmento_4',
       'segmento_5', 'segmento_6','segmento_8',
       'segmento_9', 'segmento_A', 'segmento_B', 'segmento_C',
       'segmento_G', 'segmento_M', 'segmento_S', 'segmento_otro']

df_segmentos=X_test[columnas_segmento]
df_segmentos=df_segmentos.idxmax(axis=1)
result['segmento']=df_segmentos
#result['origen']='basica'

orden_columnas=['fecha', 'segmento','real', 'predicho' , 'nombre_dia', 'fecha_2', 'day',
       'month', 'year', 'hour', 'minute']
#result.to_csv('resultado.csv')
#result.index=result['fecha']
result=result[orden_columnas]
result.sort_values('fecha',inplace=True)



result_dataframe=pd.DataFrame(columns=['fecha','segmento'])
for segmento in   lista_segmentos:
    prediccion_real=pd.DataFrame(pd.date_range(start=START_PRED,end=END_PRED, freq='0.5H', name='fecha'))
    prediccion_real['fecha']=pd.to_datetime(prediccion_real['fecha'])
    #prediccion_real=prediccion_real[prediccion_real['fecha']>='2017-11-10 08:00:00']
    #prediccion_real=prediccion_real[prediccion_real['fecha']<='2017-11-10 18:00:00']

    #creamos 2 fechas con las 8 y las 6 de la tarde para luego realizar el filtro de los horarios que no nos interesan
    hora_inicio = datetime.datetime.now()
    hora_inicio=hora_inicio.replace(hour=7, minute=30, second=0, microsecond=0)
    hora_fin = datetime.datetime.now()
    hora_fin=hora_fin.replace(hour=18, minute=0, second=0, microsecond=0)
    prediccion_real=prediccion_real[((prediccion_real['fecha'].dt.time>hora_inicio.time()) & (prediccion_real['fecha'].dt.time<hora_fin.time())) ]
    prediccion_real['segmento']=segmento
    result_dataframe=result_dataframe.append(prediccion_real)




result_dataframe

def add_missing_dummy_columns( d, columns ):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        d[c] = 0


def fix_columns( d, columns ):

    add_missing_dummy_columns( d, columns )

    # make sure we have all the columns we need
    assert( set( columns ) - set( d.columns ) == set())

    extra_cols = set( d.columns ) - set( columns )
    if extra_cols:
        print ("extra columns:"), extra_cols

    d = d[ columns ]
    return d


def fix_low_values(x):
    if(x<0):
        return 0
    else:
        return x

df_real = variables(result_dataframe)
df_real['dia_semana'] = df_real['dia_semana'].astype('str')
df_real['day'] = df_real['day'].astype('str')
df_real['month'] = df_real['month'].astype('str')
df_real['year'] = df_real['year'].astype('str')
df_real['hora'] = df_real['hora'].astype('str')
df_dummies_real = pd.get_dummies(df_real)

#ALMACENAMOS TEMPORALMENTE LAS FECHAS PARA LUEGO ASOCIARSELA AL DATASET DE PREDICCION
fechas_reales=df_dummies_real['fecha']

#UTILIZAMOS LOS MÉTODOS CREADOS PARA ELIMINAR LAS COLUMNAS ADICIONALES Y ANEXAR LAS FALTANTES CON TODOS SUS VALORES EN CEROS
test=fix_columns(df_dummies_real.copy(),X_train.columns)
df_final=pd.DataFrame(test)
#df_final.drop('total_llamadas',inplace=True,axis=1)
#SE ALMACENAN LOS SEGMENTOS ANTES DE ENTRENAR PARA VOLVER A ASIGNAR MAS ADELANTE AL RESULTADO
df_segmentos_real=df_final[columnas_segmento]
df_segmentos_real=df_segmentos_real.idxmax(axis=1)

#df_final.drop('fecha',inplace=True,axis=1)

###NOTA: LA DEFINICIÓN DE LOS SEGMNETOS SE HACE DE ACUERDO A LA PARAMETRIZACIÓN QUE TIENE MÓNICA EN SU ÁREA
def homologa_segmentos(segmento):
    if(segmento=='segmento_' or segmento=='') :
        return "otro"

    elif (segmento=='segmento_1'):

        return "EMPRESAS"
    elif (segmento=='segmento_2'):
        return "EMPRESAS"

    elif (segmento=='segmento_3'):
        return "EMPRESAS"

    elif (segmento=='segmento_4'):
        return "PERSONAS"

    elif (segmento=='segmento_5'):
        return "PYME"

    elif (segmento=='segmento_6'):
        return "PERSONAS"

    elif (segmento=='segmento_7'):
        return "EMPRESAS"

    elif (segmento=='segmento_8'):
        return "EMPRESAS"

    elif (segmento=='segmento_A'):
        return "EMPRESAS"

    elif (segmento=='segmento_B'):
        return "PYME"

    elif (segmento=='segmento_9'):
        return "PERSONAS"

    elif (segmento=='segmento_C'):
        return "EMPRESAS"

    elif (segmento=='segmento_M'):
        return "PERSONAS"

    elif (segmento=='segmento_S'):
        return "PERSONAS"

    elif (segmento=='segmento_G'):
        return "EMPRESAS"

    elif (segmento=='segmento_otro'):
        return "otro"

def cast_int(x):
    return int(x)

prediction_real = clf.predict(df_final)
prediccion_real=pd.DataFrame(prediction_real)
prediccion_real.columns=['total_llamadas']
prediccion_real['total_llamadas']=prediccion_real['total_llamadas'].apply(fix_low_values)
df_final['total_llamadas']=prediction_real
#df_final.to_excel("resultado.xlsx")
df_final['segmento']=df_segmentos_real
df_final['fecha']=fechas_reales
#df_final = variables(df_final)
#df_final=df_final['total_llamadas']
df_final['hora']=df_final['fecha'].dt.time
df_final['segmento']=df_final['segmento'].apply(homologa_segmentos)
df_final=pd.DataFrame(df_final)
df_final['fecha_llamada']=df_final['fecha'].dt.date
df_final=df_final[[ 'total_llamadas', 'segmento', 'fecha','hora','fecha_llamada']]
df_final[ 'total_llamadas']=df_final[ 'total_llamadas'].apply(cast_int)
df_final.to_excel("resultado_final.xlsx",sheet_name='PREDICCION')




# plt.figure(figsize=(15,10))
# plt.xlabel=('Fecha')
# plt.ylabel=('Numero de llamadas')
# #plt.plot(result['real'].head(100),label='real')
# #plt.plot(result['predicho'].head(100),label='predicho',lw=2,linestyle='--')
# plt.plot(result['predicho'].head(30),color='b',marker='v')
# plt.plot(result['real'].head(30),marker='o',color='g')
# plt.legend(loc='upper right')
# df_segmentos=X_test[columnas_segmento]
# df_segmentos=df_test.idxmax(axis=1)
# df_segmentos
