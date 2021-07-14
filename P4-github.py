#Luis Gabriel Masís Fernández B84666 
#I-2021 
#Universidad de Costa Rica

# Parte 4.1 de modulación con 16-QAM 
# A diferencia de la modulación BPSK, esta señal tiene dos portadoras, una que involucra seno y la otra que involucra coseno, por ende se escoge el método de la realización de dos veces la función modulación. 

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time


    ### ------------------------------------------------------------------------------------------------------------------####################
def modulador(bits, fc, mpp):
    '''Un método que simula el esquema de
    modulación digital 16-QAM.

    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La onda portadora coseno c1(t)
    :return: La onda portadora seno c2(t)
    '''
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits)  # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)  # mpp: muestras por período
    portadora_I= np.cos(2*np.pi*fc*t_periodo)
    portadora_Q = np.sin(2*np.pi*fc*t_periodo)

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp)
    senal_Tx = np.zeros(t_simulacion.shape)

    # 4. Asignar las formas de onda según los bits (16-QAM)


    # Se pone el contador j en 0
    # El valor es almacenado entre j y j+1 
    # Gracias a esto j se incrementa de forma distinta a i 
    # En el rango se utiliza 4 ya que se deben tomar de 4 en 4 bits 
    j = 0
    for i in range(0, N, 4):
        ''' Se crea la señal según los valores del 16-QAM:
        b1 = bits[i]; b2 = bits[i+1];
        b3 = bits[i+2]; b4 = bits[i+3]
        La siguiente fórmula se diseñó para compactar el código:
        senal_Tx[i*mpp: (i+1)*mpp] = (-1)**(1+b1) * 3**(1-b2) * portadora_I \
            + (-1)**(b3) * 3**(1-n4) * portadora_Q
        Con la tabla proporcionada se puede atestar que estos valores
        coinciden con los del 16-QAM.
        '''
        senal_Tx[j*mpp: (j+1)*mpp] = \
            (-1)**(1+bits[i]) * 3**(1-bits[i+1]) * portadora_I \
            + (-1)**(bits[i+2]) * 3**(1-bits[i+3]) * portadora_Q
        j += 1
    # 5. Calcular la potencia promedio de la señal modulada 
    P_senal_Tx = 1 / (N*Tc) * np.trapz(pow(senal_Tx, 2), t_simulacion)

    # Se devuelven todas las variables necesitadas luego. 
    return senal_Tx, P_senal_Tx, portadora_I, portadora_Q
        
       
        
        
     
     

def demodulador(senal_Rx, portadora_I,portadora_Q, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema BPSK. El criterio
    de demodulación se basa en decodificación por 
    detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora: La onda portadora c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits (símbolos) en transmisión
    N = int(M / mpp) 

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(senal_Rx.shape)
    

    j=0 

    # Demodulación
    for i in range(N):

        if j+4 > N:  # Los datos recuperados no siempre coinciden con los muestreados 
            break   

        
        # Producto interno de dos funciones
        producto_I = senal_Rx[i*mpp : (i+1)*mpp] * portadora_I
      

        producto_Q= senal_Rx[i*mpp : (i+1)*mpp] * portadora_Q
      

        senal_demodulada[i*mpp: (i+1)*mpp] = producto_I + producto_Q

       # Criterio de decisión por detección de energía y magnitud
        # se asumen señales en fase
    

        # Primero se detecta el signo de la amplitud
        if np.sum(producto_I) >= 0:
            bits_Rx[j] = 1  # b1 

        # Se invierte el signo
        if np.sum(producto_Q) < 0:
            bits_Rx[j+2] = 1  # b3

        #Se analiza la magnitud (utilizando el
        # punto medio entre 1 y 3) 
        if np.max(np.abs(producto_I)) < 2.5:
            bits_Rx[j+1] = 1  # b2
        if np.max(np.abs(producto_Q)) < 2.5:
            bits_Rx[j+3] = 1  # b4
        j += 4
        

    return bits_Rx.astype(int), senal_demodulada


def bits_a_rgb(bits_Rx, dimensiones):
    '''Un blque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k 
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)



# Parámetros
fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = 15 # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)

# 3. Modular la cadena de bits usando el esquema BPSK
senal_Tx, Pm, portadora_I,portadora_Q= modulador(bits_Tx, fc, mpp)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx, senal_demodulada = demodulador(senal_Rx, portadora_I,portadora_Q, mpp)

# 6. Se visualiza la imagen recibida 
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recuperado')
Fig.tight_layout()
plt.imshow(imagen_Rx)

#Es importante destacar que la calidad de la imagen recuperada va a depender de la SNR utilizada, en este caso, si se sube la SNR un poco más no se darían errores, pero se utiliza una SNR que muestre que se pueden dar errores aunque sean mínimos. 


# Visualizar el cambio entre las señales
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(14, 7))

# La onda cuadrada moduladora (bits de entrada)
ax1.plot(moduladora[0:600], color='r', lw=2) 
ax1.set_ylabel('$b(t)$')

# La señal modulada por BPSK
ax2.plot(senal_Tx[0:600], color='g', lw=2) 
ax2.set_ylabel('$s(t)$')

# La señal modulada al dejar el canal
ax3.plot(senal_Rx[0:600], color='b', lw=2) 
ax3.set_ylabel('$s(t) + n(t)$')

# La señal demodulada
ax4.plot(senal_demodulada[0:600], color='m', lw=2) 
ax4.set_ylabel('$b^{\prime}(t)$')
ax4.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()

# Pregunta 4.2: Estacionaridad y Ergocidad 

# Los tiempos son definidos 
T = 100
t_f= 0.1
t = np.linspace(0, t_f, T)
    
# Inicialización del proceso aleatorio X(t) con N realizaciones
A = [-3, -1, 1, 3]            # Se tiene dos posibles valores para la amplitud
N =  16                       # Hay 4 posibilidades de la onda
General_t = np.empty((N, len(t)))    # N funciones del tiempo x(t) con T puntos

# Para la señal se aplica cada valor de amplitud (-1 y 1)
for i in A:
    H = i*np.cos(2*np.pi*fc*t) + i*np.sin(2*np.pi*fc*t)
    I = -i*np.cos(2*np.pi*fc*t) + i*np.sin(2*np.pi*fc*t)
    J = i*np.cos(2*np.pi*fc*t) + -i*np.sin(2*np.pi*fc*t)
    K = -i*np.cos(2*np.pi*fc*t) + -i*np.sin(2*np.pi*fc*t)
    General_t[i,:]   = H
    General_t[i+1,:] = I
    General_t[i+2,:] = J
    General_t[i+3,:] = K
    plt.plot(t, H)
    plt.plot(t, I)
    plt.plot(t, J)
    plt.plot(t, K)

# Promedio de las N realizaciones en cada instante (cada punto en t)

P = [np.mean(General_t[:,i]) for i in range(len(t))]
Rea, = plt.plot(t, P, lw=6, color = 'blue', label='Promedio realizaciones')

# Graficar el resultado teórico del valor esperado

E = np.mean(senal_Tx)*t
Teo, = plt.plot(t, E, '-.', lw=3, color = 'r', label='Valor teorico')

    # Mostrar las realizaciones, y su promedio calculado y teórico
plt.title('Realizaciones del proceso aleatorio $X(t)$')
plt.xlabel('$t$')
plt.ylabel('$x_i(t)$')
plt.legend([Rea, Teo], ['Promedio realizaciones', 'Valor teorico'])
plt.show()   

# T valores de desplazamiento tau
desplazamiento = np.arange(T)
taus = desplazamiento/t_f

# Inicialización de matriz de valores de correlación para las N funciones
corr = np.empty((N, len(desplazamiento)))

# Nueva figura para la autocorrelación
plt.figure()

# Cálculo de correlación para cada valor de tau
for n in range(N):
    for i, tau in enumerate(desplazamiento):
        corr[n, i] = np.correlate(General_t[n,:], np.roll(General_t[n,:], tau))/T
    plt.plot(taus, corr[n,:])

# Valor teórico de correlación donde la magnitud es (varianza - media^2) y ya que es una variable de Bernoulli con p = 0.5
Rxx = (0.5 - np.power(0.25,2)) * np.cos(np.pi*taus)

# Gráficas de correlación para cada realización y la
plt.plot(taus, Rxx, '-.', lw=4, label='Correlación teórica')
plt.title('Funciones de autocorrelación de las realizaciones del proceso')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$R_{WW}(\tau)$')
plt.legend()
plt.show()

#4.3 Densidad espectral de potencia 

from scipy.fft import fft

# Transformada de Fourier
senal_f = fft(senal_Tx)

# Muestras de la señal
Nm = len(senal_Tx)

# Número de símbolos (198 x 89 x 8 x 3)
Ns = Nm // mpp

# Tiempo del símbolo = periodo de la onda portadora
Tc = 1 / fc

# Tiempo entre muestras (período de muestreo)
Tm = Tc / mpp

# Tiempo de la simulación
T = Ns * Tc

# Espacio de frecuencias
f = np.linspace(0.0, 1.0/(2.0*Tm), Nm//2)

# Gráfica
plt.plot(f, 2.0/Nm * np.power(np.abs(senal_f[0:Nm//2]), 2))
plt.xlim(0, 20000)
plt.grid()
plt.show()