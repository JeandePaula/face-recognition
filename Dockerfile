# 1. Imagem Base com Python
FROM python:3.9-slim

# 2. Definir Variáveis de Ambiente (Opcional, mas útil)
ENV PYTHONUNBUFFERED=1
# Garante que prints apareçam imediatamente no log do Docker

# 3. Instalar Dependências do Sistema Operacional
# Necessário para compilar dlib (dependência do face_recognition) e para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Adicione libgl1-mesa-glx se precisar rodar com GUI (requer config extra no 'docker run')
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Criar e Definir o Diretório de Trabalho dentro do container
WORKDIR /app

# 5. Copiar o arquivo de dependências
COPY requirements.txt .

# 6. Instalar as Dependências Python
# Usar --no-cache-dir para manter a imagem menor
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copiar o restante dos arquivos do projeto (script e imagens)
COPY . .

# 8. Comando para rodar a aplicação quando o container iniciar
# Rodará o script main.py usando python
# A variável DISABLE_GUI pode ser passada no comando 'docker run'
CMD ["python", "main.py"]