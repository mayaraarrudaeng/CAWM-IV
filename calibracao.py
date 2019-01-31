import pandas as pd
import numpy as np
import math
from statistics import mean
import scipy.optimize as opt

frame = pd.DataFrame()
path_par = r'dados\parametros_de_entrada.csv'
parametros = pd.read_csv(path_par,float_precision='high')

A = parametros.loc[0,'Valores']
#estimativas iniciais das variáveis de decisão
x0 = np.zeros(3)
x0[0]= parametros.loc[1,'Valores'] #Ks
x0[1]= parametros.loc[3,'Valores'] #p
x0[2]= parametros.loc[4,'Valores'] #SUBmax

b = parametros.loc[2,'Valores'] #b = 5/3 Parâmetro não calibrável
T = parametros.loc[6,'Valores'] # intervalo de tempo em segundos
n = parametros.loc[7,'Valores'] #n
c = parametros.loc[8,'Valores'] #c
Be = parametros.loc[9,'Valores'] #B - largura equivalente
Lt = parametros.loc[10,'Valores'] #Comprimento do rio principal até o ponto estudado
I = parametros.loc[11,'Valores'] #Inclinação do rio
a = parametros.loc[12,'Valores'] #<--multiplicador da taxa de evapotranspiração a extrair do solo

k = (T/n)*((c**2*A**2/Be**2/Lt**5)**(1/3))*I**0.5

#Ler os resultados estatisticos
est = pd.read_csv(r'resultados\Resultados_estatisticos.csv')

#Ler o intervalo correspondente ao Nash máximo
n_inter = est.loc[est['Nash'] == max(est['Nash']),'Unnamed: 0']

#Ler o arquivo em que se encontra o período
periodo = pd.read_csv(r'dados\intervalo.csv')

#Ler o arquivo em que se encontra o balanço hídrico sem calibração
CAWM = pd.read_csv(r'resultados\CAWM_IV.csv')

def obj_function(x):
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
    def RE(x):
        E = (1 - math.exp(-a*(frame.loc[i,'reserv_solo']/x[2])))*frame.loc[i,'evap_n_atendida']
        return min(frame.loc[i,'evap_n_atendida'],frame.loc[i,'reserv_solo'],E)

    def C():
        if (frame.loc[i,'S1']<=0):
            return 0
        else:
            return min(k*frame.loc[i,'S1']**b,frame.loc[i,'S1'])
    def Ps(x):
        Pn = frame.loc[i,'chuva_media']-frame.loc[i,'evap_inicial']
        hiperb = np.tanh(Pn/x[2])
        Sub = frame.loc[i,'reserv_solo']/x[2]
        termo1 = x[2]*(1-Sub**2)*hiperb
        termo2 = 1+Sub*hiperb
        Ps = termo1/termo2
        return max(Ps,0)
    def rec_rio(x):
        return x[0] * frame.loc[i, 'Solo']

    def zerolistmaker(n):
        listofzeros = [float(0)] * n
        return listofzeros
    def solo(frame):
        return max(frame.loc[i, 'reserv_solo']-frame.loc[i, 'RE'], 0)
    def perdas(x):
        return min(x[1] * (frame.loc[i, 'S2'] ** 1.5), frame.loc[i, 'S2'])


    intervalo1 = int(periodo.loc[n_inter - 1, 'inicio'])-1
    intervalo2 = int(periodo.loc[n_inter - 1, 'fim'])
    #intervalo3 = intervalo2 - 1

    obs = CAWM.loc[intervalo1:intervalo2, 'vazao']
    chuva = CAWM.loc[intervalo1:intervalo2, 'chuva_media']
    evap = CAWM.loc[intervalo1:intervalo2, 'evaporacao']

    frame = pd.concat([obs, chuva, evap], axis=1)

    frame['evap_inicial'] = ""
    frame.loc[intervalo1,'evap_inicial'] = CAWM.loc[intervalo1,'evap_inicial']
    frame['retencao'] = ""
    frame.loc[intervalo1,'retencao'] = CAWM.loc[intervalo1,'retencao']
    frame['evap_n_atendida'] = ""
    frame.loc[intervalo1,'evap_n_atendida'] = CAWM.loc[intervalo1,'evap_n_atendida']
    frame['ret_corrig'] = ""
    frame.loc[intervalo1,'ret_corrig'] = CAWM.loc[intervalo1,'ret_corrig']
    frame['escoamento'] = ""
    frame.loc[intervalo1,'escoamento'] = CAWM.loc[intervalo1,'escoamento']
    frame['reserv_solo'] = ""
    frame.loc[intervalo1,'reserv_solo'] = CAWM.loc[intervalo1,'reserv_solo']
    frame['S1'] = ""
    frame.loc[intervalo1,'S1'] = CAWM.loc[intervalo1,'S1']
    frame['RE'] = ""
    frame.loc[intervalo1,'RE'] = CAWM.loc[intervalo1,'RE']
    frame['Solo'] = ""
    frame.loc[intervalo1,'Solo'] = CAWM.loc[intervalo1,'Solo']
    frame['C'] = ""
    frame.loc[intervalo1,'C'] = CAWM.loc[intervalo1,'C']
    frame['S2'] = ""
    frame.loc[intervalo1,'S2'] = CAWM.loc[intervalo1,'S2']
    frame['vazao_calc'] = ""
    frame.loc[intervalo1,'vazao_calc'] = CAWM.loc[intervalo1,'vazao_calc']
    frame['Ps'] = ""
    frame.loc[intervalo1,'Ps'] = CAWM.loc[intervalo1,'Ps']
    frame['rec_solo'] = ""
    frame.loc[intervalo1,'rec_solo'] = CAWM.loc[intervalo1,'rec_solo']
    frame['rec_rio'] = ""
    frame.loc[intervalo1,'rec_rio'] = CAWM.loc[intervalo1,'rec_rio']
    frame['perdas'] = ""
    frame.loc[intervalo1,'perdas'] = CAWM.loc[intervalo1,'perdas']
    frame['S4'] = ""
    frame.loc[intervalo1,'S4'] = CAWM.loc[intervalo1,'S4']
    frame['reserv_solo_corrig'] = ""
    frame.loc[intervalo1,'reserv_solo_corrig'] = CAWM.loc[intervalo1,'reserv_solo_corrig']

    for i in range(intervalo1+1,intervalo2+1):
        frame.loc[i,'evap_inicial'] = evap_inicial(frame)
        frame.loc[i,'retencao'] = max(frame.loc[i-1,'ret_corrig']+frame.loc[i,'chuva_media']-frame.loc[i,'evap_inicial'],0)
        frame.loc[i,'evap_n_atendida'] = frame.loc[i,'evaporacao']-frame.loc[i,'evap_inicial']
        frame.loc[i,'ret_corrig'] = ret_corrig()
        frame.loc[i, 'reserv_solo'] = reserv_solo()
        frame.loc[i, 'Ps'] = Ps(x)
        frame.loc[i, 'rec_solo'] = frame.loc[i, 'Ps']
        frame.loc[i,'escoamento'] = frame.loc[i,'chuva_media']-frame.loc[i,'evap_inicial']-frame.loc[i,'rec_solo']
        frame.loc[i, 'RE'] = RE(x)
        frame.loc[i, 'Solo'] = solo(frame)
        frame.loc[i, 'rec_rio'] = rec_rio(x)
        frame.loc[i, 'S1'] = frame.loc[i - 1, 'S4'] + frame.loc[i, 'escoamento'] + frame.loc[i, 'rec_rio']
        frame.loc[i,'C'] = C()
        frame.loc[i,'S2'] =frame.loc[i,'S1']-frame.loc[i,'C']
        frame.loc[i,'vazao_calc'] = (frame.loc[i,'C']/1000)*(A*1000000/86400)
        frame.loc[i,'perdas'] = perdas(x)
        frame.loc[i,'S4'] = frame.loc[i,'S2'] - frame.loc[i,'perdas']
        frame.loc[i,'reserv_solo_corrig'] = reserv_solo_corrig()

    s = frame.loc[intervalo1+1:intervalo2+1,'vazao_calc']
    o = obs.loc[intervalo1+1:intervalo2+1]
    vazao_obs = CAWM.loc[1:,'vazao']
    vazao_obs_corrig = vazao_obs.dropna()
    o_med = mean(vazao_obs_corrig)
    Nash = NS(s, o)

    simulado = frame.loc[intervalo1+1:intervalo2+1,'vazao_calc']
    observado = obs.loc[intervalo1+1:intervalo2+1]
    s_med = mean(simulado)
    o_media = mean(observado.dropna())
    observado1 = observado.replace(np.nan, 0, regex=True)
    Rsqr = Rsquared()

    sim = frame.loc[intervalo1+1:intervalo2+1,'vazao_calc']
    obs = obs.loc[intervalo1+1:intervalo2+1]
    obs = obs.replace(np.nan, 0, regex=True)
    soma_abs= sum(abs(s-o))

    obj_function = (soma_abs/Nash)*1000000
    return obj_function

b1 = (0, 1)
b2 = (10, None)
bnds = (b1, b1, b2)

final_obj_function = opt.minimize(fun=obj_function, x0=x0, method='TNC', bounds=bnds)

print(final_obj_function)

file = open("resultados/otimizacao.txt","w")
file.write(str(final_obj_function))
file.close()
