"""
Actividad formativa 2
Simulación y análisis de señales con la transformada de Fourier

En este programa hago lo siguiente:
  1. Creo tres señales básicas: pulso rectangular, escalón y senoidal.
  2. Les saco la transformada de Fourier con np.fft.fft()
  3. Grafico la magnitud y la fase del espectro de cada una.
  4. Compruebo tres propiedades: linealidad, desplazamiento en el tiempo
     y escalamiento.
  5. Hago un ejemplo "real": quitarle el ruido de 60 Hz a una señal.

Librerías: numpy y matplotlib (con eso alcanza).
Las gráficas se guardan en la carpeta "figuras".
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # para que guarde las imágenes sin abrir ventanas
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 0. Datos generales
# ---------------------------------------------------------------
fs = 1000                      # frecuencia de muestreo (muestras por segundo)
N = 2000                       # número de muestras (2 segundos en total)

# Uso n = -1000 ... 999 para que el tiempo 0 quede justo en la mitad.
n = np.arange(-N // 2, N // 2)
t = n / fs                     # eje del tiempo, de -1 s a 0.999 s

# Eje de frecuencias. fftshift lo acomoda de negativo a positivo.
f = np.fft.fftshift(np.fft.fftfreq(N, d=1 / fs))

os.makedirs("figuras", exist_ok=True)


def calcular_fft(x):
    """
    Calcula la transformada de Fourier de x.
    Antes uso ifftshift para que el tiempo 0 (la mitad del arreglo) quede
    al principio, que es lo que espera la fft de numpy. Después uso
    fftshift para que el espectro quede centrado en 0 Hz.
    Divido entre N para que las amplitudes sean más fáciles de leer.
    """
    X = np.fft.fft(np.fft.ifftshift(x)) / N
    return np.fft.fftshift(X)


def fase_limpia(X):
    """
    Saca la fase (en radianes). Donde la magnitud es casi cero la fase sale
    con valores raros por errores de redondeo, entonces ahí la pongo en 0.
    """
    fase = np.angle(X)
    fase[np.abs(X) < 1e-4 * np.max(np.abs(X))] = 0
    return fase


# ---------------------------------------------------------------
# 1. Definición de las señales en el dominio del tiempo
# ---------------------------------------------------------------
def pulso(mitad_ancho, desplazamiento=0):
    """Pulso rectangular de altura 1. Dura (2*mitad_ancho + 1) muestras."""
    return (np.abs(n - desplazamiento) <= mitad_ancho).astype(float)


x_pulso = pulso(100)                          # dura más o menos 0.2 s
x_escalon = (t >= 0).astype(float)            # vale 0 antes de t=0 y 1 después
x_seno = np.sin(2 * np.pi * 50 * t)           # senoidal de 50 Hz, amplitud 1

# Gráfica de las tres señales en el tiempo
fig, ejes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
datos = [(x_pulso, "Pulso rectangular"),
         (x_escalon, "Función escalón"),
         (x_seno, "Senoidal de 50 Hz")]
for ax, (x, nombre) in zip(ejes, datos):
    ax.plot(t, x, color="tab:blue")
    ax.set_title(nombre)
    ax.set_ylabel("x(t)")
    ax.grid(alpha=0.3)
ejes[2].set_xlim(-0.5, 0.5)
ejes[2].set_xlabel("Tiempo (s)")
plt.tight_layout()
plt.savefig("figuras/fig1_senales_tiempo.png", dpi=130)
plt.close()


# ---------------------------------------------------------------
# 2. Cálculo de la transformada y gráficas (magnitud y fase)
# ---------------------------------------------------------------
def graficar_senal_y_espectro(x, nombre, archivo, lim_t, lim_f):
    """Dibuja la señal, la magnitud y la fase de su transformada."""
    X = calcular_fft(x)
    fig, ejes = plt.subplots(3, 1, figsize=(8, 8))

    ejes[0].plot(t, x, color="tab:blue")
    ejes[0].set_title(f"{nombre} en el tiempo")
    ejes[0].set_xlabel("Tiempo (s)")
    ejes[0].set_xlim(*lim_t)
    ejes[0].grid(alpha=0.3)

    ejes[1].plot(f, np.abs(X), color="tab:red")
    ejes[1].set_title("Magnitud del espectro")
    ejes[1].set_xlabel("Frecuencia (Hz)")
    ejes[1].set_xlim(*lim_f)
    ejes[1].grid(alpha=0.3)

    ejes[2].plot(f, fase_limpia(X), color="tab:green")
    ejes[2].set_title("Fase del espectro")
    ejes[2].set_xlabel("Frecuencia (Hz)")
    ejes[2].set_ylabel("rad")
    ejes[2].set_xlim(*lim_f)
    ejes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(archivo, dpi=130)
    plt.close()
    return X


X_pulso = graficar_senal_y_espectro(x_pulso, "Pulso rectangular",
                                    "figuras/fig2_pulso.png", (-0.5, 0.5), (-40, 40))
X_escalon = graficar_senal_y_espectro(x_escalon, "Escalón",
                                      "figuras/fig3_escalon.png", (-0.5, 0.5), (-40, 40))
X_seno = graficar_senal_y_espectro(x_seno, "Senoidal de 50 Hz",
                                   "figuras/fig4_seno.png", (-0.1, 0.1), (-100, 100))

# Datos para fijarme en los resultados
i50 = np.argmin(np.abs(f - 50))
print("---- Resultados de los espectros ----")
print(f"Senoidal: el pico está en +-{abs(f[np.argmax(np.abs(X_seno))]):.0f} Hz, "
      f"magnitud {np.abs(X_seno[i50]):.3f} (esperado 0.5 = A/2)")
print(f"Senoidal: fase en +50 Hz = {np.angle(X_seno[i50]):.3f} rad (esperado -pi/2 = -1.571)")
print(f"Pulso: valor en 0 Hz = {np.abs(X_pulso[N // 2]):.4f} "
      f"(es la suma de las muestras entre N = {x_pulso.sum() / N:.4f})")
print(f"Escalón: valor en 0 Hz = {np.abs(X_escalon[N // 2]):.4f}")


# ---------------------------------------------------------------
# 3. Propiedades de la transformada de Fourier
# ---------------------------------------------------------------
print("\n---- Propiedades ----")

# 3.1 Linealidad: FFT(a*x1 + b*x2) debe ser igual a a*FFT(x1) + b*FFT(x2)
a, b = 2.0, 0.5
x_suma = a * x_pulso + b * x_seno
X_suma_directa = calcular_fft(x_suma)
X_suma_por_separado = a * X_pulso + b * X_seno
dif_lineal = np.max(np.abs(X_suma_directa - X_suma_por_separado))
print(f"Linealidad: diferencia máxima = {dif_lineal:.2e}  -> "
      f"{'se cumple' if dif_lineal < 1e-12 else 'NO se cumple'}")

fig, ejes = plt.subplots(3, 1, figsize=(8, 8))
ejes[0].plot(t, x_suma, color="tab:blue")
ejes[0].set_title("x(t) = 2·pulso(t) + 0.5·seno(t)")
ejes[0].set_xlim(-0.3, 0.3)
ejes[0].set_xlabel("Tiempo (s)")
ejes[0].grid(alpha=0.3)
ejes[1].plot(f, np.abs(X_suma_directa), color="tab:red", label="FFT de la suma")
ejes[1].plot(f, np.abs(X_suma_por_separado), "k--", label="2·FFT(pulso) + 0.5·FFT(seno)")
ejes[1].set_title("Magnitud: las dos curvas coinciden")
ejes[1].set_xlim(-80, 80)
ejes[1].set_xlabel("Frecuencia (Hz)")
ejes[1].legend(fontsize=8)
ejes[1].grid(alpha=0.3)
ejes[2].plot(f, np.abs(X_suma_directa - X_suma_por_separado), color="tab:purple")
ejes[2].set_title("Diferencia entre las dos (es prácticamente cero)")
ejes[2].set_xlim(-80, 80)
ejes[2].set_xlabel("Frecuencia (Hz)")
ejes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figuras/fig5_linealidad.png", dpi=130)
plt.close()

# 3.2 Desplazamiento en el tiempo: si x(t) se retrasa t0, la magnitud no cambia
#     y la fase cambia en -2*pi*f*t0
desp = 250                       # 250 muestras = 0.25 s
t0 = desp / fs
x_pulso_desp = pulso(100, desplazamiento=desp)
X_desp = calcular_fft(x_pulso_desp)

mag_igual = np.allclose(np.abs(X_desp), np.abs(X_pulso), atol=1e-12)
X_esperado = X_pulso * np.exp(-1j * 2 * np.pi * f * t0)
dif_desp = np.max(np.abs(X_desp - X_esperado))
print(f"Desplazamiento: magnitudes iguales = {mag_igual}; "
      f"diferencia con X·e^(-j2πft0) = {dif_desp:.2e}")

fig, ejes = plt.subplots(3, 1, figsize=(8, 8))
ejes[0].plot(t, x_pulso, label="Original", color="tab:blue")
ejes[0].plot(t, x_pulso_desp, label=f"Retrasado {t0} s", color="tab:orange")
ejes[0].set_title("Pulso original y pulso desplazado")
ejes[0].set_xlim(-0.5, 0.8)
ejes[0].set_xlabel("Tiempo (s)")
ejes[0].legend(fontsize=8)
ejes[0].grid(alpha=0.3)
ejes[1].plot(f, np.abs(X_pulso), label="Original", color="tab:blue", lw=3, alpha=0.5)
ejes[1].plot(f, np.abs(X_desp), "--", label="Desplazado", color="tab:orange")
ejes[1].set_title("Magnitud: queda igual")
ejes[1].set_xlim(-40, 40)
ejes[1].set_xlabel("Frecuencia (Hz)")
ejes[1].legend(fontsize=8)
ejes[1].grid(alpha=0.3)
ejes[2].plot(f, fase_limpia(X_pulso), label="Original", color="tab:blue")
ejes[2].plot(f, fase_limpia(X_desp), label="Desplazado", color="tab:orange")
ejes[2].set_title("Fase: cambia (es una recta que se envuelve entre -π y π)")
ejes[2].set_xlim(-40, 40)
ejes[2].set_xlabel("Frecuencia (Hz)")
ejes[2].set_ylabel("rad")
ejes[2].legend(fontsize=8)
ejes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figuras/fig6_desplazamiento.png", dpi=130)
plt.close()

# 3.3 Escalamiento: si el pulso se hace la mitad de angosto en el tiempo,
#     su espectro se hace el doble de ancho en frecuencia
x_ancho = pulso(100)             # más o menos 0.2 s
x_angosto = pulso(50)            # más o menos 0.1 s
X_ancho = calcular_fft(x_ancho)
X_angosto = calcular_fft(x_angosto)


def primer_cero(X):
    """Primera frecuencia positiva donde la magnitud baja de 2% del máximo."""
    mag = np.abs(X)
    positivas = f > 0
    indices = np.where(positivas & (mag < 0.02 * mag.max()))[0]
    return f[indices[0]]


fc_ancho = primer_cero(X_ancho)
fc_angosto = primer_cero(X_angosto)
print(f"Escalamiento: pulso de 0.2 s -> primer cero en {fc_ancho} Hz; "
      f"pulso de 0.1 s -> primer cero en {fc_angosto} Hz "
      f"(el ancho en frecuencia se multiplicó por {fc_angosto / fc_ancho:.2f})")

fig, ejes = plt.subplots(2, 2, figsize=(10, 6))
ejes[0, 0].plot(t, x_ancho, color="tab:blue")
ejes[0, 0].set_title("Pulso de 0.2 s")
ejes[0, 1].plot(f, np.abs(X_ancho), color="tab:red")
ejes[0, 1].set_title(f"Su espectro (primer cero ≈ {fc_ancho:.0f} Hz)")
ejes[1, 0].plot(t, x_angosto, color="tab:blue")
ejes[1, 0].set_title("Pulso de 0.1 s")
ejes[1, 1].plot(f, np.abs(X_angosto), color="tab:red")
ejes[1, 1].set_title(f"Su espectro (primer cero ≈ {fc_angosto:.0f} Hz)")
for i in range(2):
    ejes[i, 0].set_xlim(-0.4, 0.4)
    ejes[i, 0].set_xlabel("Tiempo (s)")
    ejes[i, 1].set_xlim(-60, 60)
    ejes[i, 1].set_xlabel("Frecuencia (Hz)")
    ejes[i, 1].set_ylim(0, 0.11)
    for j in range(2):
        ejes[i, j].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figuras/fig7_escalamiento.png", dpi=130)
plt.close()


# ---------------------------------------------------------------
# 4. Aplicación real: quitar el ruido de la red eléctrica (60 Hz)
# ---------------------------------------------------------------
# Imagino un sensor que mide una señal lenta de 5 Hz (por ejemplo una señal
# del cuerpo, tipo latido o movimiento). El cable mete ruido de 60 Hz de la
# corriente de la casa y además hay un poco de ruido aleatorio.
rng = np.random.default_rng(0)
senal_buena = np.sin(2 * np.pi * 5 * t)
ruido_60 = 0.5 * np.sin(2 * np.pi * 60 * t)
ruido_azar = 0.2 * rng.standard_normal(N)
senal_medida = senal_buena + ruido_60 + ruido_azar

X_med = calcular_fft(senal_medida)

# Filtro en frecuencia: dejo solo lo que esté entre -20 y 20 Hz y el resto lo hago cero
X_filtrada = X_med.copy()
X_filtrada[np.abs(f) > 20] = 0
# Regreso al tiempo con la transformada inversa
senal_limpia = np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(X_filtrada) * N)))

error_antes = np.sqrt(np.mean((senal_medida - senal_buena) ** 2))
error_despues = np.sqrt(np.mean((senal_limpia - senal_buena) ** 2))
print("\n---- Aplicación ----")
print(f"Error (RMS) de la señal medida:   {error_antes:.3f}")
print(f"Error (RMS) de la señal filtrada: {error_despues:.3f}")

fig, ejes = plt.subplots(2, 2, figsize=(11, 7))
ejes[0, 0].plot(t, senal_medida, color="gray", lw=0.8)
ejes[0, 0].set_title("Señal medida (con ruido)")
ejes[0, 1].plot(f, np.abs(X_med), color="tab:red")
ejes[0, 1].set_title("Espectro: pico de 5 Hz y pico de 60 Hz")
ejes[1, 0].plot(t, senal_limpia, color="tab:green", label="Filtrada")
ejes[1, 0].plot(t, senal_buena, "k--", lw=0.8, label="Señal original")
ejes[1, 0].set_title("Señal después del filtro")
ejes[1, 0].legend(fontsize=8)
ejes[1, 1].plot(f, np.abs(X_filtrada), color="tab:green")
ejes[1, 1].set_title("Espectro después de quitar > 20 Hz")
for j in range(2):
    ejes[j, 0].set_xlim(-0.5, 0.5)
    ejes[j, 0].set_xlabel("Tiempo (s)")
    ejes[j, 1].set_xlim(-100, 100)
    ejes[j, 1].set_xlabel("Frecuencia (Hz)")
    ejes[j, 1].set_ylim(0, 0.6)
    for i in range(2):
        ejes[j, i].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figuras/fig8_aplicacion.png", dpi=130)
plt.close()

print("\nListo, las gráficas están en la carpeta 'figuras'.")
