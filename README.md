[README.md](https://github.com/user-attachments/files/33368242/README.md)# Simulación y análisis de señales con la transformada de Fourier

Actividad formativa 2. En este proyecto uso Python para ver tres señales básicas
(pulso rectangular, escalón y senoidal) en el tiempo y en la frecuencia, usando la
transformada de Fourier.

## ¿Qué hace el código?

1. Crea las señales: pulso rectangular, escalón y senoidal de 50 Hz.
2. Calcula su transformada.
3. Grafica la magnitud y la fase del espectro de cada una.
4. Comprueba tres propiedades:
   - Linealidad
   - Desplazamiento en el tiempo
   - Escalamiento
5. Ejemplo real: quita el ruido de 60 Hz de una señal de 5 Hz filtrando en frecuencia.

## Archivos

- `fourier_senales.py`: código con comentarios.
- `figuras/`: gráficas generadas.

## Resultados

- La senoidal de 50 Hz da dos picos en ±50 Hz con magnitud 0.5.
- El pulso da un espectro tipo sinc con primer cero en 5 Hz.
- Las tres propiedades se cumplen (diferencias del orden de 1e-17).
- Al filtrar el ruido de 60 Hz, el error bajó de 0.411 a 0.039.
