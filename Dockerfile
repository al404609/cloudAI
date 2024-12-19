# Usa una imagen base con Python
FROM python:3.9-slim

# Configura el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos necesarios al contenedor
COPY requirements.txt .
COPY server.py .

# Copia los directorios necesarios dentro del contenedor
COPY results_article /app/results_article
COPY saved_model_article /app/saved_model_article
COPY saved_model_legal /app/saved_model_legal

# Instala las dependencias
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Expone el puerto donde correrá la aplicación Flask
EXPOSE 5320

# Define el comando para ejecutar la aplicación
CMD ["python", "server.py"]
