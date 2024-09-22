import cv2
import os # manipular diretórios
import numpy as np
import openpyxl # criar/manipular Excel
import matplotlib.pyplot as plt

# encontra a posição dos três retângulos principais na imagem
def contornosPrincipais(imagem):
    frame = imagem.copy()
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imagem_limiarizada = cv2.threshold(frame_cinza, 200, 255, cv2.THRESH_BINARY_INV)[1]
    contornos, hierarquia = cv2.findContours(imagem_limiarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = []

    for contorno in contornos:
        (x, y, w, h) = cv2.boundingRect(contorno)

        area = int(w) * int(h)

        if area > 10000:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            areas.append((area, x, y, w, h))

    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # plt.title("contorno principal")
    # plt.show()

    return sorted(areas)

# Verifica se a prova está alinhada, senão é ajustada
def ajustaProva(imagem):
    # O 'x' do maior contorno da imagem sempre está abaixo de 50 quando a imagem está normal (sem inclinação)
    while True:
        areas = contornosPrincipais(imagem)
        if areas[-1][1] <= areas[0][1] and areas[-1][2] > areas[0][2]:
            break
        else:
            # Rotacionar 90 graus para a direita
            imagem = cv2.rotate(imagem, cv2.ROTATE_90_CLOCKWISE)

    return imagem

# Identifica o tipo da prova
def versaoProva(imagem):
    frame = imagem.copy()
    # Processamento de imagem para melhor identificação do contorno
    imagem_escalaCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imagem_suavizada = cv2.GaussianBlur(imagem_escalaCinza, (7, 7), 0)
    imagem_limiarizada_marcado = cv2.threshold(imagem_suavizada, 120, 255, cv2.THRESH_BINARY)[1]

    contornosMarcados, hierarquiaMarcados = cv2.findContours(imagem_limiarizada_marcado, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    opcao = []

    # Identifica o contorno que corresponde ao tipo da prova
    for contorno in contornosMarcados:
        (x_axis,y_axis), raio = cv2.minEnclosingCircle(contorno)

        centro = (int(x_axis),int(y_axis))
        raio = int(raio)
        cv2.circle(frame, centro, raio, (0, 255, 0), 2)

        area = 3.14 * (raio * raio)

        if area > 200 and area < 850:
            opcao.append(centro)
            break

    opcao = max([i[0] for i in opcao])

    # Identifica qual o tipo da prova baseado na posição onde o contorno está
    if opcao < 200:
        opcao = 'A'
    elif opcao < 400:
        opcao = 'B'
    elif opcao < 600:
        opcao = 'C'
    elif opcao < 900:
        opcao = 'D'
    else: opcao = 'E'
    
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # plt.title(opcao)
    # plt.show()

    return opcao

# Encontra as opções selecionadas e as opções desmarcadas
def encontraContornos(imagem):
    # Processamento de imagem para melhor identificação dos contornos
    imagem_escalaCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_suavizada = cv2.GaussianBlur(imagem_escalaCinza, (7, 7), 0)
    imagem_limiarizada_marcados = cv2.threshold(imagem_suavizada, 100, 255, cv2.THRESH_BINARY_INV)[1]
    imagem_limiarizada_geral = cv2.threshold(imagem_escalaCinza, 230, 255, cv2.THRESH_BINARY_INV)[1]

    # Identifica somente o contorno das opções multipla-escolha marcadas
    contornosMarcados, hierarquiaMarcados = cv2.findContours(imagem_limiarizada_marcados, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Identifica todos os contornos das opções multipla-escolha
    contornosGeral, hierarquiaGeral = cv2.findContours(imagem_limiarizada_geral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Contém somente os contornos das opções multipla-escolha marcadas
    listaMarcados = []
    # Contém somente os contornos das opções multipla-escolha
    listaGeral = []

    # Aproxima os contornos de opções marcadas para contornos circulares
    for contorno in contornosMarcados:
        (x_axis,y_axis), raio = cv2.minEnclosingCircle(contorno) 

        centro = (int(x_axis),int(y_axis)) 
        raio = int(raio) 

        area = 3.14 * (raio * raio)
        if area > 200:
            listaMarcados.append(centro)

    # Aproxima todos contornos de opções para contornos circulares
    for contorno in contornosGeral:
        (x_axis,y_axis), raio = cv2.minEnclosingCircle(contorno) 

        centro = (int(x_axis),int(y_axis)) 
        raio = int(raio) 

        area = 3.14 * (raio * raio)
        if area > 500 and area < 900:
            listaGeral.append(centro)

    '''
    Daqui para baixo são apenas códigos destinados a realizar uma 'limpeza'
    e filtragem nos contornos, de modo a ficar apenas aqueles que correspondem
    às opções (selecionadas ou não).
    '''

    # Acrescenta à lista de contornos gerais as opções marcadas
    listaGeral += listaMarcados

    # Ordena os contornos gerais pela altura
    listaGeral = sorted(listaGeral, key=lambda x: x[1])

    # Agrupa os elementos de mesma altura
    alturaMax = listaGeral[0][1]
    listaGeralAgrupada = []
    vetorAux = []
    for i in listaGeral:
        # Somente elementos com altura de até 3 pixels de diferença
        if abs(i[1] - alturaMax) > 3:
            listaGeralAgrupada.append(vetorAux)
            vetorAux = []
        vetorAux.append(i)
        alturaMax = i[1]
    listaGeralAgrupada.append(vetorAux)

    # Ordena a matriz pelo comprimento
    for i in listaGeralAgrupada: i.sort()

    # Mantém na matriz somente elementos com tamanho maior que 5
    listaGeral = [sorted(i) for i in listaGeralAgrupada if len(i) >= 5]

    # Limpa a matriz de elementos iguais ou muito parecidos
    matrizLimpa = []
    for i in listaGeral:
        if i:
            vetorAux = []
            aux = i[0][0]
            vetorAux.append(i[0])
            for j in i:
                if (j not in vetorAux):
                    if abs(aux - j[0]) > 15:
                        vetorAux.append(j)
                        aux = j[0]
            matrizLimpa.append(vetorAux)

    listaGeral = matrizLimpa

    # Agrupa os elementos de mesmo comprimento
    comprimentoMax = listaGeral[0][0][0]
    listaGeralAgrupada = []
    vetorAux = []
    for i in listaGeral:
        for j in i:
            # Somente elementos com comprimento de até 3 pixels de diferença
            if abs(j[0] - comprimentoMax) > 40:
                listaGeralAgrupada.append(vetorAux)
                vetorAux = []
            vetorAux.append(j)
            comprimentoMax = j[0]
    listaGeralAgrupada.append(vetorAux)

    # Mantém na matriz somente elementos com tamanho igual a 5
    listaGeral = [sorted(i) for i in listaGeralAgrupada if len(i) >= 5]

    # Ordena por comprimento
    listaGeral = sorted(listaGeral)

    # Agrupa a cada 25 elementos (1-25, 26-50, 51-75, 76-100)
    matriz = []
    aux = []
    i = 1
    for j in listaGeral:
        aux.append(j)
        if i == 25:
            matriz.append(aux)
            aux = []
            i = 0
        i += 1
    
    listaGeral = matriz

    # Ordena pela altura
    matriz = []
    for i in listaGeral:
        matriz.append(sorted(i, key=lambda x: x[0][1]))

    listaGeral = matriz

    # Retira o agrupamento por 25 (1-25, 26-50, 51-75, 76-100)
    matriz = []
    for i in listaGeral:
        for j in i:
            matriz.append(j)

    listaGeral = matriz

    # for i in listaGeral:
    #     for j in i:
    #         cv2.circle(imagem, j, 10, (0, 255, 0), 2)
    
    # plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    # plt.show()
            
    return listaGeral, listaMarcados

# Retorna uma matriz contendo as opções multipla-escolha selecionadas
def processarFrame(imagem):
    # Cria um vetor que guardará a opção selecionada para cada questão
    escolhas = np.zeros((100, 5))
    # Índice = número da questão
    indice = 0

    # Somente a parte de interesse é analisada
    mainContorno = contornosPrincipais(imagem)[-1]
    frame = imagem[mainContorno[2] - 20:mainContorno[2] + mainContorno[4] + 20]
    #frame = imagem[int(imagem.shape[0]*0.25):imagem.shape[0], :,:].copy()

    # Lista de contende a posição das opções selecionadas e as opções desmarcadas
    listaGeral, listaMarcados = encontraContornos(frame)

    # Identifica a opção marcada
    for i in listaGeral:
        for j in range(len(i)):
            for k in listaMarcados:
                diff = (abs(i[j][0] - k[0]), abs(i[j][1] - k[1]))
                if  diff[0] < 10 and diff[1] < 10:
                    escolhas[indice][j] = 1
    
        indice += 1

    contadorBranco = []

#   Mostra a matriz com as opções escolhidas
    for i in range(1, 31):
        if not any(escolhas[i - 1]):
            # print(f"{i}:Branco!")
            contadorBranco.append(i)
        # print(f"{i}: {escolhas[i - 1]}")
    

    return [escolhas[0:30][:], contadorBranco] # nessa prova o aluno só escolhe até 30

# Permite o usuário criar o próprio gabarito
def gabaritoUsuario():
    # Vetor que guarda as opções marcadas pelo usuário
    marcados = []

    # Função que é chamada a cada evento feito pelo mouse
    def evento_mouse(log, x, y, amp, stop):
        if log == cv2.EVENT_LBUTTONDOWN:
            # Verifica se a posição clicada é um contorno que corresponde a uma opção
            for i in listaGeral:
                for j in i:
                    if j[1] <= y + 6 and j[1] >= y - 6:
                        if j[0] <= x + 6 and j[0] >= x - 6:
                            indice = listaGeral.index(i)
                            # Se a opção já foi marcada então ela é retirada
                            if (j[0], j[1], indice) in marcados:
                                cv2.circle(frame, (j[0], j[1]), 7, (255, 255, 255), -1)
                                marcados.remove((j[0], j[1], indice))
                            else:
                                # Caso já tenha uma opção marcada na mesma questão ele apenas troca                                
                                if indice in [k[2] for k in marcados]:
                                    # Descobre o índice da opção que já estava marcada
                                    indiceAux = [k[2] for k in marcados].index(indice)
                                    marcadoAux = (marcados[indiceAux][0], marcados[indiceAux][1], indice)
                                    # Retira a opção já marcada e marca outra
                                    cv2.circle(frame, (marcadoAux[0], marcadoAux[1]), 7, (255, 255, 255), -1)
                                    cv2.circle(frame, (j[0], j[1]), 7, (0, 0, 0), -1)
                                    marcados.remove((marcadoAux[0], marcadoAux[1], indice))    
                                else:
                                    # Marca uma opção nova 
                                    cv2.circle(frame, (j[0], j[1]), 7, (0, 0, 0), -1)
                                marcados.append((j[0], j[1], indice))
            for i in listaBotao:
                if abs(x - i[0]) <= 2 and abs(y - i[1]) <= 2:
                    stop.append(False)
                    break
    gabaritoUser = cv2.imread(f'{pathProvas}\\Cartao_resposta_v2.jpg')
    
    #mainContorno = contornosPrincipais(gabaritoUser)[-1]
    # frame = gabaritoUser[mainContorno[2] - 20:mainContorno[2] + mainContorno[4] + 20]
    frame = gabaritoUser[int(gabaritoUser.shape[0]*0.28):gabaritoUser.shape[0],:,:].copy()

    listaGeral, listaMarcados = encontraContornos(frame)

    '''
    A imagem estava muito grande então tivemos que redimensioná-la
    Com isso o código para encontrar contornos acabou tendo problemas
    Mas ainda utilizamos ele só que agora multiplicamos cada valor
    dos contornos encontrados por essa escala abaixo que relaciona
    o tamanho da imagem com o de outra imagem menor (1080, 720)
    '''
    escala_x = 1080/frame.shape[1]
    escala_y = 720/frame.shape[0]

    # Servirá para parar as iterações quando o usuário clicar no botão 'confirmar'
    stop = []

    for i in range(len(listaGeral)):
        for j in range(len(listaGeral[i])):
            listaGeral[i][j] = (int(listaGeral[i][j][0] * escala_x), int(listaGeral[i][j][1] * (escala_y - 0.17) + 202))
    
    # Permite o usuário marcar as opções da prova
    cv2.namedWindow('Gabarito Usuario')
    cv2.setMouseCallback('Gabarito Usuario', evento_mouse, stop)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Diminui o tamanho da imagem para ficar melhor para o usuário selecionar as opções
    frame = gabaritoUser
    frame = cv2.resize(frame, (1080, 720))

    # Acrescenta as opções do tipo de prova (em 1080, 720) à lista de contornos gerais
    listaOpcao = [(55, 132), (276, 132), (488, 132), (706, 132), (927, 132)]
    listaGeral.append(listaOpcao)

    # Adiciona um botão de confirmação na imagem
    cv2.rectangle(frame, (frame.shape[1]//2 - 100, frame.shape[0] - 80), (frame.shape[1]//2 + 100, frame.shape[0] - 40), (0, 255, 0), -1)
    cv2.rectangle(frame, (frame.shape[1]//2 - 100, frame.shape[0] - 80), (frame.shape[1]//2 + 100, frame.shape[0] - 40), (0, 0, 0), 2)
    cv2.putText(frame, 'Confirmar', (frame.shape[1]//2 - 80, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Posição do botão
    listaBotao = []
    for i in range(440, 650, 2):
        for j in range(640, 680, 2):
            listaBotao.append((i, j))

    # Deixa a imagem aberta até o usuário apertar 'ESC' ou até o usuário clicar em confirmar
    while all(stop):
        cv2.imshow('Gabarito Usuario', frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break
        # Não deixa o usuário confirmar até que ele marque a opção da prova
        if not all(stop):
            try:
                frameAux = frame.copy()
                frameAux = cv2.resize(frameAux, (gabaritoUser.shape[1], gabaritoUser.shape[0]))
                versaoProva(frameAux[int(frameAux.shape[0]*0.16):int(frameAux.shape[0]*0.2), :,:])
            except:
                print('ruim')
                stop.remove(False)
                cv2.putText(frame, "Marque o tipo da prova!", ((frame.shape[1]//2 - 80, 60)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.destroyAllWindows()

    # Retira o botão de confirmar da imagem
    cv2.rectangle(frame, (frame.shape[1]//2 - 120, frame.shape[0] - 85), (frame.shape[1]//2 + 120, frame.shape[0] - 30), (255, 255, 255), -1)

    # Retira o aviso caso o usuário esqueça de marcar o tipo da prova
    cv2.rectangle(frame, (450, 30), (850, 66), (255, 255, 255), -1)

    # Marca as opções na prova de tamanho original para não perder a qualidade de resolução
    for i in range(len(marcados)):
        marcados[i] = (int(marcados[i][0] / escala_x), int(marcados[i][1] / (escala_y - 0.17)))

    for i in marcados:
        cv2.circle(gabaritoUser, i, 15, (0, 0, 0), -1)

    return gabaritoUser

# Cria um novo arquivo Excel
workbook = openpyxl.Workbook()
# Seleciona a planilha ativa
sheet = workbook.active

# Caminho no diretório contendo os arquivos
pathProvas = (os.path.abspath("C:")) + "\\Leitor"

# Limpa a pasta que contém os gabaritos do usuário
for filename in os.listdir(f'{pathProvas}\\gabarito-usuario'):
    if filename:
        remover = input(f"Foi identificado um gabarito-usuario: '{filename}'. Você deseja removê-lo? (1) - Sim | (0) - Não: ")
        if int(remover) == 1:
            os.remove(f'{pathProvas}\\gabarito-usuario\\{filename}')

# Cria o gabarito do usuário
while True:
    if int(input("Você deseja criar um gabarito? (1) - Sim | (0) - Não: ")):
        gabaritoUser = gabaritoUsuario()
        try:
            mainContorno = contornosPrincipais(gabaritoUser)[0]
            frame = gabaritoUser[mainContorno[2] - 20:mainContorno[2] + mainContorno[4] + 20]
            opcao = versaoProva(frame)

            cv2.imwrite(f'{pathProvas}\\gabarito-usuario\\Gabarito{opcao}.jpg', gabaritoUser)
            print("\n--- Gabarito criado com sucesso! ---\n")
        except:
            print("\n--- Falha na criação do Gabarito! ---\n")
    else: break

# Identifica os arquivos das provas
for filename in os.listdir(f'{pathProvas}\\provas realizadas'):
    print(f"Processando arquivo {filename}...")
    cartao_resposta = cv2.imread(f'{pathProvas}\\provas realizadas\\{filename}')
    cartao_resposta = ajustaProva(cartao_resposta)
    [aluno_resposta, brancos] = processarFrame(cartao_resposta)

    # Descobre o tipo da prova, se não conseguir seta para tipo A
    try:
        versionContorno = contornosPrincipais(cartao_resposta)[0]
        opcao = versaoProva(cartao_resposta[versionContorno[2] - 20:versionContorno[2] + versionContorno[4] + 20])
    except:
        opcao = 'A'

    # opcao = 'A'

    gabaritoAux = 0
    # Verifica se há um gabarito feito pelo usuário para essa prova, se houver então ele será utilizado
    for gabaritoUser in os.listdir(f'{pathProvas}\\gabarito-usuario'):
        if gabaritoUser == f'Gabarito{opcao}.jpg':
            gabarito_resposta = cv2.imread(f'{pathProvas}\\gabarito-usuario\\{gabaritoUser}')
            gabarito = processarFrame(gabarito_resposta)[0]
            gabaritoAux = 1
    # Caso contrário identifica um gabarito padrão
    if not gabaritoAux:
        for gabarito in os.listdir(f'{pathProvas}\\prova {opcao}'):
            gabarito_resposta = cv2.imread(f'{pathProvas}\\prova {opcao}\\{gabarito}')
            gabarito = processarFrame(gabarito_resposta)[0]

    # Matriz booleana das resposta do aluno com a do gabarito
    comparacao = (aluno_resposta == gabarito)
    corretas = 0
    erradas = []

    for i in range(len(comparacao)):
        if all(comparacao[i]): corretas += 1
        else: erradas.append(i + 1)

    # Escreve os dados numa linha do Excel
    linhaExcel = [f'Aluno:'] + [f'{filename}']
    linhaExcel += ['Nota:'] + [corretas] + ['Taxa de Acerto:'] + [f'{int((corretas/30)*100)}%']
    linhaExcel += ['Questões incorretas (incluindo em branco):'] + [str(erradas)]
    linhaExcel += ['Em brancos: '] + [str(brancos)]

    # Acrescenta a linha na planilha
    sheet.append(linhaExcel)

# Formata as colunas no excel para comportar o conteúdo
for coluna in sheet.columns:
    maxLargura = 0
    colunaLetra = coluna[0].column_letter  # Pega a letra da coluna
    for celula in coluna:
        if len(str(celula.value)) > maxLargura:
            maxLargura = len(str(celula.value))
    sheet.column_dimensions[colunaLetra].width = maxLargura + 2


print("Arquivos processados")
# Salva o arquivo Excel
workbook.save(pathProvas + "\\notas\\notas.xlsx")