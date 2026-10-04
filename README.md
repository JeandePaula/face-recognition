# Face Camera

Reconhecimento facial local para câmeras IP, webcams e arquivos de vídeo. Detecta
rostos com **YuNet**, compara com fotos cadastradas usando **SFace** e salva o
quadro original quando uma pessoa aparece. Funciona com janela ou sem interface.

A versão 2 foi reconstruída com Python 3.14, OpenCV 5 e NumPy 2. Apenas duas
dependências de execução; sem `dlib`, compilação C++, servidor ou banco de dados.
Python 3.12 ou superior é suportado. Versões exatas ficam no `uv.lock`.

## Começar no WSL/Linux

Instale o [uv](https://docs.astral.sh/uv/getting-started/installation/) e as
bibliotecas de sistema do OpenCV:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
git clone https://github.com/JeandePaula/face-recognition.git
cd face-recognition
uv sync --locked
uv run face-camera download-models
cp config.example.toml config.toml
mkdir -p known_faces
# Adicione fotos antes de reconhecer pessoas (veja abaixo).
uv run face-camera run --source 0
```

O `uv` instala automaticamente o Python definido em `.python-version`. No WSL,
câmeras USB precisam estar disponíveis no Linux; uma URL de câmera IP ou um
arquivo de vídeo também funciona. Para instalação com pip, use um ambiente
virtual Python >=3.12 e `pip install .` (o lock é aplicado pelo uv).

```bash
uv run face-camera run --source 'http://camera:8080/video'
uv run face-camera run --source ./video.mp4 --max-frames 100
uv run face-camera run --show # Q fecha a janela; precisa de WSLg/X11/display
```

## Cadastrar pessoas

Use fotos de pessoas que autorizaram o uso, com **exatamente um rosto** visível.
O nome vem do arquivo ou da pasta. Várias fotos da mesma pessoa são aceitas:

```text
known_faces/
├── Ana.jpg
└── João/
    ├── frente.png
    └── lateral.jpg
```

São aceitos JPG, JPEG, PNG, WebP e BMP. Reinicie o programa após alterar o
cadastro. Fotos ilegíveis, sem rosto ou com vários rostos interrompem a carga
com uma mensagem, evitando cadastrar uma pessoa errada. A pasta vazia permite
detecção apenas: todos os rostos aparecem como `Unknown`.

## Configuração

Copie `config.example.toml` para `config.toml`, carregado automaticamente, ou use
`face-camera run --config /caminho/config.toml`. Caminhos relativos são resolvidos
a partir da pasta do arquivo de configuração. Sem arquivo, usa a pasta atual.

| Opção | Padrão | Uso |
| --- | --- | --- |
| `source` | `0` | Índice da webcam (inteiro), URL HTTP/RTSP ou vídeo |
| `known_faces_dir` | `known_faces` | Fotos de referência |
| `models_dir` | `models` | Modelos ONNX verificados |
| `output_dir` | `detected_faces_output` | Quadros salvos em JPEG |
| `headless` | `true` | Executar sem janela |
| `resize_factor` | `0.5` | Escala da detecção, maior que 0 e até 1 |
| `detection_threshold` | `0.9` | Confiança mínima do detector |
| `match_threshold` | `0.363` | Similaridade mínima; maior é mais rigoroso |
| `snapshot_cooldown` | `10.0` | Intervalo mínimo em segundos por nome |
| `reconnect_attempts` | `5` | Tentativas após falha de câmera |
| `reconnect_delay` | `2.0` | Segundos entre tentativas |
| `timeout_ms` | `5000` | Timeout de abertura/leitura de URLs via FFmpeg |

Precedência: argumentos CLI > variáveis de ambiente > TOML > padrões.
`FACE_SOURCE` define a fonte sem gravar credenciais no arquivo; `IP_CAMERA_URL`
é aceito como alternativa legada. `DISABLE_GUI=true|false` também é compatível.
O programa não carrega `.env` automaticamente.

Um nome continuamente visível gera uma foto na chegada. Ao desaparecer e voltar,
gera outra após o intervalo configurado. Desconhecidos compartilham um único
grupo `Unknown`: **não há rastreamento nem identificação individual deles**.
Cada snapshot contém o quadro completo, sem anotações. Nomes são sanitizados;
arquivos recebem horário UTC e identificador aleatório. Falhas de escrita geram
erro e código de saída diferente de zero.

Câmeras reconectam até o limite; uma leitura bem-sucedida reinicia a contagem.
Vídeos terminam no fim do arquivo sem reconectar. Ctrl+C e SIGTERM liberam a
captura e fecham a janela. O timeout configurável é para URLs; drivers de webcam
podem ter seus próprios limites de bloqueio.

## Modelos e limites

`download-models` baixa cerca de 39 MB do OpenCV Zoo, com revisão fixa e SHA-256,
usando arquivo temporário e substituição atômica. Arquivos já válidos são
reutilizados. A execução verifica os modelos e **não faz downloads implícitos**.
Depois do download, a inferência funciona offline e as imagens ficam locais.
Para outro diretório, use `download-models --models-dir /caminho/models` e ajuste
o TOML para o mesmo local.

O limiar inicial segue o [exemplo oficial do OpenCV](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html).
Calibre-o com imagens representativas da sua câmera; similaridade não é uma
probabilidade. Iluminação, pose e tamanho do rosto afetam o resultado. Este
projeto não implementa prova de vida nem autenticação segura. Os modelos foram
treinados para rostos humanos; desenhos antigos do repositório não são cadastros
adequados para esse pipeline.

## Docker

```bash
docker build -t face-camera .
mkdir -p models known_faces detected_faces_output
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/models:/app/models" \
  face-camera download-models
docker run --rm --user "$(id -u):$(id -g)" \
  -e FACE_SOURCE='http://camera:8080/video' \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/known_faces:/app/known_faces:ro" \
  -v "$PWD/detected_faces_output:/app/detected_faces_output" \
  face-camera run --headless
```

A imagem executa como usuário não privilegiado. `--user` permite escrever em
pastas do host pertencentes ao seu usuário. Para configuração própria, monte
também `config.toml` em `/app/config.toml:ro`. Fotos, configurações locais e
modelos não são copiados para a imagem nem versionados. A saída não tem retenção
automática; acompanhe o espaço em disco e exclua snapshots quando necessário.

## Desenvolvimento

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

Os testes de unidade não precisam de câmera nem rede. Com modelos baixados,
`uv run pytest -m integration` também executa inferência real usando uma imagem
sintética sem rosto. O teste com rosto real é opcional: configure
`FACE_TEST_IMAGE=/caminho/foto-com-um-rosto.jpg uv run pytest -m integration`.
Não inclua fotos pessoais de teste no Git.

```text
src/face_camera/
├── cli.py          # comandos e ciclo de execução
├── config.py       # configuração validada
├── models.py       # download e integridade
├── recognition.py  # detecção, cadastro e comparação
├── camera.py       # leitura, reconexão e liberação
└── snapshots.py    # controle e gravação de imagens
```

A CI verifica estilo, testes em Python 3.12/3.13/3.14, modelos reais, pacote e
imagem Docker. Para atualizar dependências: ajuste os limites no `pyproject.toml`
quando necessário, rode `uv lock --upgrade` e repita os testes.

### Migração da versão 1

`config.py`, `face_utils.py`, `image_saver.py` e `requirements.txt` foram
substituídos pelo pacote e pelo TOML. `uv run python main.py` continua sendo um
atalho para `face-camera run`. Mova imagens cadastradas para `known_faces` e
configure a fonte: não há mais endereço privado embutido. Os embeddings mudaram
de dlib para SFace; `TOLERANCE` antigo não se aplica ao novo limiar. Cadastros são
recalculados na inicialização. Exemplos de desenhos e snapshots antigos foram
retirados da árvore atual; continuam no histórico Git.

## Licença

Código sob [MIT](LICENSE). YuNet usa MIT e SFace usa Apache-2.0; consulte
[THIRD_PARTY.md](THIRD_PARTY.md) e `licenses/` para as licenças dos modelos.
