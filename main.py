import cv2
import numpy as np
import face_recognition
import time
import datetime # Importado para timestamp
import os       # Importado para criar diretório

# --- Configurações ---
IP_CAMERA_URL = "http://192.168.1.66:8080/video"  # URL da sua câmera IP
KNOWN_FACES_DATA = [
    ("Pessoa 1", "Morty.png"), # Certifique-se que este arquivo existe
    # ("Nome da Pessoa 2", "caminho/para/imagem2.jpg"),
]
RESIZE_FACTOR = 0.5
TOLERANCE = 0.6
OUTPUT_DIR = "detected_faces_output" # Diretório para salvar as imagens

# --- Função para Carregar Faces Conhecidas ---
# (Nenhuma alteração necessária nesta função)
def load_known_faces(known_faces_data: list) -> tuple[list, list]:
    """Carrega codificações de faces e nomes a partir de arquivos de imagem."""
    known_face_encodings = []
    known_face_names = []
    print("Carregando faces conhecidas...")
    for name, image_path in known_faces_data:
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) == 1:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"  [OK] Rosto de '{name}' carregado de '{image_path}'")
            elif len(encodings) > 1:
                print(f"  [AVISO] Mais de um rosto encontrado em '{image_path}'. Usando o primeiro.")
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
            else:
                print(f"  [ERRO] Nenhum rosto encontrado em '{image_path}'. Imagem ignorada.")
        except FileNotFoundError:
            print(f"  [ERRO] Arquivo de imagem não encontrado em '{image_path}'. Verifique o caminho.")
        except Exception as e:
            print(f"  [ERRO] Ocorreu um erro ao carregar '{image_path}': {e}")
    if not known_face_names:
        print("\nAVISO: Nenhuma face conhecida foi carregada com sucesso.")
    else:
        print(f"\nCarregamento concluído. {len(known_face_names)} face(s) conhecida(s) pronta(s).")
    return known_face_encodings, known_face_names

# --- Função para Processar um Único Frame ---
# Modificada para retornar também o conjunto de nomes detectados
def process_frame(frame: np.ndarray, known_face_encodings: list, known_face_names: list) -> tuple[np.ndarray, set]:
    """Detecta e reconhece faces em um frame de vídeo e retorna o frame anotado e um conjunto de nomes detectados."""

    detected_names_in_frame = set() # Conjunto para armazenar nomes neste frame
    processed_frame = frame.copy() # Trabalha com uma cópia para desenhar

    # 1. Redimensiona (se necessário)
    if RESIZE_FACTOR != 1.0:
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    else:
        small_frame = processed_frame # Usa a cópia se não redimensionar

    # 2. Converte BGR -> RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 3. Encontra faces e codificações
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # 4. Itera sobre cada face encontrada
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
        name = "Desconhecido"
        color = (0, 0, 255)

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                color = (0, 255, 0)
                # print(f"Status: Pessoa Reconhecida - {name}") # Mantido para debug se quiser
            # else:
                # print("Status: Pessoa Detectada - Desconhecida") # Mantido para debug se quiser

        # Adiciona o nome (conhecido ou "Desconhecido") ao conjunto do frame atual
        detected_names_in_frame.add(name)

        # 5. Re-escala coordenadas
        top, right, bottom, left = face_location
        if RESIZE_FACTOR != 1.0:
            top = int(top / RESIZE_FACTOR)
            right = int(right / RESIZE_FACTOR)
            bottom = int(bottom / RESIZE_FACTOR)
            left = int(left / RESIZE_FACTOR)

        # 6. Desenha no frame *processado*
        cv2.rectangle(processed_frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(processed_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(processed_frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    # Retorna o frame com desenhos E o conjunto de nomes detectados nele
    return processed_frame, detected_names_in_frame

# --- Função Principal ---
def main():
    # Carrega as faces conhecidas
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DATA)

    # Cria o diretório de saída se não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Imagens de novas detecções serão salvas em: '{OUTPUT_DIR}'")

    # Inicializa a captura de vídeo
    print(f"\nTentando conectar à câmera IP: {IP_CAMERA_URL}...")
    cap = cv2.VideoCapture(IP_CAMERA_URL)

    if not cap.isOpened():
        print("-" * 40)
        print(f"ERRO CRÍTICO: Não foi possível abrir o stream de vídeo.")
        # ... (mensagens de erro detalhadas) ...
        print("-" * 40)
        return

    print("Câmera conectada com sucesso! Iniciando o reconhecimento...")
    # Adicione esta linha se for rodar com GUI (se não, ignore ou comente)
    # print("Pressione 'q' na janela de vídeo para sair.")

    # Variável para rastrear faces vistas no *frame anterior*
    previous_faces_seen = set()

    # Verifica se a GUI deve ser desabilitada (útil para Docker sem display)
    DISABLE_GUI = os.environ.get('DISABLE_GUI', 'false').lower() == 'true'
    if DISABLE_GUI:
        print("Modo sem GUI ativado (via variável de ambiente DISABLE_GUI).")
    else:
        print("Pressione 'q' na janela de vídeo para sair.")


    while True:
        # Captura frame por frame (guardamos o original)
        ret, original_frame = cap.read()

        if not ret or original_frame is None:
            print("Erro ao capturar frame ou stream finalizado.")
            # ... (lógica de reconexão) ...
            # (código de reconexão omitido para brevidade, mantenha o seu se necessário)
            print("Tentando reconectar em 5 segundos...") # Exemplo simples
            time.sleep(5)
            # Tentar reabrir aqui ou quebrar o loop
            # Para simplificar, vamos quebrar por agora:
            break # Sai do loop se não conseguir ler o frame

        # Processa o frame (recebe o frame anotado e o conjunto de nomes atuais)
        processed_frame, current_faces_seen = process_frame(
            original_frame, known_encodings, known_names
        )

        # --- Lógica para Salvar Imagem em Nova Detecção ---
        # Compara o conjunto atual com o anterior para encontrar novas aparições
        newly_appeared_faces = current_faces_seen - previous_faces_seen

        if newly_appeared_faces:
            # Gera um timestamp único para este momento de detecção
            # Inclui microssegundos para evitar colisões se várias faces aparecerem no mesmo segundo
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            # Itera sobre cada *nome distinto* que acabou de aparecer
            for name in newly_appeared_faces:
                if name == "Desconhecido":
                    prefix = "person_unknown"
                    print(f"Status: Nova pessoa DESCONHECIDA detectada.")
                else:
                    prefix = "person_known"
                    print(f"Status: Nova detecção da pessoa CONHECIDA: {name}")

                # Monta o nome do arquivo
                filename = os.path.join(OUTPUT_DIR, f"{prefix}-{timestamp}.png")

                # Salva o frame *original* (sem as anotações)
                try:
                    cv2.imwrite(filename, original_frame)
                    print(f"  >> Imagem salva: {filename}")
                except Exception as e:
                    print(f"  [ERRO] Falha ao salvar a imagem {filename}: {e}")

        # Atualiza o conjunto de faces vistas para a próxima iteração
        previous_faces_seen = current_faces_seen
        # --- Fim da Lógica de Salvar Imagem ---

        # Mostra o frame resultante (APENAS se a GUI não estiver desabilitada)
        if not DISABLE_GUI:
            cv2.imshow('Reconhecimento Facial (IP Camera) - Pressione Q para sair', processed_frame)

            # Verifica a tecla 'q' (APENAS se a GUI estiver ativa)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Tecla 'q' pressionada. Encerrando...")
                break

    # Libera recursos
    cap.release()
    if not DISABLE_GUI:
        cv2.destroyAllWindows()
    print("Recursos liberados. Programa finalizado.")

# --- Ponto de Entrada ---
if __name__ == '__main__':
    main()