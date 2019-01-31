import pandas as pd
import csv
import numpy as np
import math
import matplotlib.pyplot as plt

print("Bem-vindo ao CAWM versão 4.0 ")

path_par = r'dados\parametros_de_entrada.csv'
parametros = pd.read_csv(path_par,float_precision='high')

A = parametros.loc[0,'Valores']

Ks = parametros.loc[1,'Valores'] #Ks
p = parametros.loc[3,'Valores'] #p
SUBmax = parametros.loc[4,'Valores'] #SUBmax
b = parametros.loc[2,'Valores'] #b = 5/3 Parâmetro não calibrável
T = parametros.loc[6,'Valores'] # intervalo de tempo em segundos
n = parametros.loc[7,'Valores'] #n
c = parametros.loc[8,'Valores'] #c
Be = parametros.loc[9,'Valores'] #B - largura equivalente
Lt = parametros.loc[10,'Valores'] #Comprimento do rio principal até o ponto estudado
I = parametros.loc[11,'Valores'] #Inclinação do rio
a = parametros.loc[12,'Valores'] #<--multiplicador da taxa de evapotranspiração a extrair do solo

k = (T/n)*((c**2*A**2/Be**2/Lt**5)**(1/3))*I**0.5
print(k)

def filter_nan(sim, obs):
    if sim.isnull().sum() >= 1 or obs.isnull().sum() >= 1:
        obs.isnull()
        obs.fillna(0, inplace=True)
        sim.isnull()
        sim.fillna(0, inplace=True)
        return sim, obs
    else:
        pass

def NS(sim, obs):
    filter_nan(sim, obs)
    return 1 - sum((sim - obs) ** 2) / sum((obs - o_med) ** 2)

def Rsquared():

    f_termo = sum((simulado - s_med) * (observado1 - o_media))
    s_termo = (sum((simulado - s_med) ** 2) * sum((observado1 - o_media) ** 2)) ** (1 / 2)
    rquad = (f_termo / s_termo) ** 2
    return rquad

def evap_inicial(frame):
    if (frame.loc[i-1,'ret_corrig']+frame.loc[i,'chuva_media']>= frame.loc[i,'evaporacao']):
        return frame.loc[i, 'evaporacao']
    else:
        return frame.loc[i-1,'ret_corrig']+frame.loc[i,'chuva_media']
def ret_corrig():
    if(frame.loc[i,'retencao']> 0):
        return 0
    else:
        return frame.loc[i,'retencao']
def reserv_solo():
    if i == 1:
        return parametros.loc[5,'Valores']
    else:
        return frame.loc[i-1,'reserv_solo_corrig']
def reserv_solo_corrig():
    return max(frame.loc[i,'Solo']+frame.loc[i,'rec_solo']-frame.loc[i,'rec_rio'],0)
def RE():
    E = (1 - math.exp(-a*(frame.loc[i,'reserv_solo']/SUBmax)))*frame.loc[i,'evap_n_atendida']
    return min(frame.loc[i,'evap_n_atendida'],frame.loc[i,'reserv_solo'],E)

def C():
    if (frame.loc[i,'S1']<=0):
        return 0
    else:
        return min(k*frame.loc[i,'S1']**b,frame.loc[i,'S1'])
def Ps():
    Pn = frame.loc[i,'chuva_media']-frame.loc[i,'evap_inicial']
    hiperb = np.tanh(Pn/SUBmax)
    Sub = frame.loc[i,'reserv_solo']/SUBmax
    termo1 = SUBmax*(1-Sub**2)*hiperb
    termo2 = 1+Sub*hiperb
    Ps = termo1/termo2
    return max(Ps,0)
def rec_rio():
    return Ks * frame.loc[i, 'Solo']

def zerolistmaker(n):
    listofzeros = [float(0)] * n
    return listofzeros
def solo(frame):
    return max(frame.loc[i, 'reserv_solo']-frame.loc[i, 'RE'], 0)
def perdas():
    return min(p * (frame.loc[i, 'S2'] ** 1.5), frame.loc[i, 'S2'])

frame = pd.DataFrame()
#caminho do arquivo de vazao
path_vazao =r'dados\vazao.csv'
#caminho do arquivo de chuva
path_chuva =r'dados\precipitacao.csv'
#caminho do arquivo de evaporação
path_evap =r'dados\evaporacao.csv'
diretorio_resultados =r'resultados'

#cria um DataFrame vazio
#lendo o arquivo de vazao
vazao = pd.read_csv(path_vazao,float_precision='high')
#lendo o arquivo de chuva
chuva = pd.read_csv(path_chuva, usecols=['chuva_media'],float_precision='high')
#concatena as colunas de vazao e chuva em um mesmo dataframe
frame = pd.concat([vazao, chuva],axis=1)
frame['mes'] = pd.DatetimeIndex(frame['data']).month

#cria um dicionario com o arquivo de evaporação
with open('dados\evaporacao.csv', mode='r') as infile:
    reader = csv.reader(infile)
    mydict = {int(rows[0]):float(rows[1]) for rows in reader}

#adiciona uma nova coluna no dataframe com os valores de evaporação
frame['evaporacao'] = frame['mes'].map(mydict)

n = len(frame.columns) #numero de colunas do dataframe

frame.loc[-1] = zerolistmaker(n)
frame.index = frame.index+1
frame = frame.sort_index()

frame['evap_inicial'] = ""
frame.loc[0,'evap_inicial'] = float(0)
frame['retencao'] = ""
frame.loc[0,'retencao'] = float(0)
frame['evap_n_atendida'] = ""
frame.loc[0,'evap_n_atendida'] = float(0)
frame['ret_corrig'] = ""
frame.loc[0,'ret_corrig'] = float(0)
frame['escoamento'] = ""
frame.loc[0,'escoamento'] = float(0)
frame['reserv_solo'] = ""
frame.loc[0,'reserv_solo'] = float(0)
frame['S1'] = ""
frame.loc[0,'S1'] = float(0)
frame['RE'] = ""
frame.loc[0,'RE'] = float(0)
frame['Solo'] = ""
frame.loc[0,'Solo'] = float(0)
frame['C'] = ""
frame.loc[0,'C'] = float(0)
frame['S2'] = ""
frame.loc[0,'S2'] = float(0)
frame['vazao_calc'] = ""
frame.loc[0,'vazao_calc'] = float(0)
frame['Ps'] = ""
frame.loc[0,'Ps'] = float(0)
frame['rec_solo'] = ""
frame.loc[0,'rec_solo'] = float(0)
frame['rec_rio'] = ""
frame.loc[0,'rec_rio'] = float(0)
frame['perdas'] = ""
frame.loc[0,'perdas'] = float(0)
frame['S4'] = ""
frame.loc[0,'S4'] = float(0)
frame['reserv_solo_corrig'] = ""
frame.loc[0,'reserv_solo_corrig'] = float(0)
for i in range(1,len(frame)):
    frame.loc[i,'evap_inicial'] = evap_inicial(frame)
    frame.loc[i,'retencao'] = max(frame.loc[i-1,'ret_corrig']+frame.loc[i,'chuva_media']-frame.loc[i,'evap_inicial'],0)
    frame.loc[i,'evap_n_atendida'] = frame.loc[i,'evaporacao']-frame.loc[i,'evap_inicial']
    frame.loc[i,'ret_corrig'] = ret_corrig()
    frame.loc[i, 'reserv_solo'] = reserv_solo()
    frame.loc[i, 'Ps'] = Ps()
    frame.loc[i, 'rec_solo'] = frame.loc[i, 'Ps']
    frame.loc[i,'escoamento'] = frame.loc[i,'chuva_media']-frame.loc[i,'evap_inicial']-frame.loc[i,'rec_solo']
    frame.loc[i, 'RE'] = RE()
    frame.loc[i, 'Solo'] = solo(frame)
    frame.loc[i, 'rec_rio'] = rec_rio()
    frame.loc[i, 'S1'] = frame.loc[i - 1, 'S4'] + frame.loc[i, 'escoamento'] + frame.loc[i, 'rec_rio']
    frame.loc[i,'C'] = C()
    frame.loc[i,'S2'] =frame.loc[i,'S1']-frame.loc[i,'C']
    frame.loc[i,'vazao_calc'] = (frame.loc[i,'C']/1000)*(A*1000000/86400)
    frame.loc[i,'perdas'] = perdas()
    frame.loc[i,'S4'] = frame.loc[i,'S2'] - frame.loc[i,'perdas']
    frame.loc[i,'reserv_solo_corrig'] = reserv_solo_corrig()

frame.to_csv(diretorio_resultados+'/CAWM_IV.csv', float_format='%.12f')

print('Arquivo criado.')

chuva2 = -1*frame['chuva_media']

fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(frame.index,frame['vazao'],label='Qobs m³/s',color='red')
lns2 = ax.plot(frame.index,frame['vazao_calc'],label='Qcalc m³/s',color='blue')
ax2 = ax.twinx()
lns3 = ax2.plot(frame.index,chuva2,label='chuva',color='green')

lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax.legend(lns,labs,loc = 0)

#Estabelece os intervalos dos eixos
ax2.set_yticks(np.arange(-1000,0,100))
ax.set_yticks(np.arange(0,450,50))
ax.set_xticks(np.arange(0,len(frame),500))

#Insere nomes nos eixos
ax.set_xlabel('Dias corridos')
ax.set_ylabel('Vazao')
ax2.set_ylabel('Chuva')

#Insere um título no gráfico
plt.title('CAWM_IV: Período Total')
plt.show()#Mostrando gráfico
plt.savefig(diretorio_resultados+'/Gráfico_período_total.png', bbox_inches='tight')