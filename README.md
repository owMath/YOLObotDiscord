# Bot de Discord para Detecção YOLO

Este é um bot de Discord que utiliza o modelo YOLO (You Only Look Once) para detectar objetos em imagens.

## Funcionalidades

- Detecta objetos em imagens usando o modelo YOLOv8
- Retorna uma imagem com os objetos marcados
- Lista os objetos detectados e suas quantidades

## Requisitos

- Python 3.8 ou superior
- Bibliotecas listadas em `requirements.txt`
- Token de bot do Discord

## Configuração

1. Clone este repositório:
```
git clone <url-do-repositorio>
cd <nome-da-pasta>
```

2. Instale as dependências:
```
pip install -r requirements.txt
```

3. O modelo YOLOv8 será baixado automaticamente na primeira execução

4. Configure o token do Discord:
   - Crie um arquivo `config.env` baseado no `config.env.example`
   - Adicione seu token do Discord no arquivo `config.env`

5. Execute o bot:
```
python bot.py
```

## Comandos do Bot

- `!detect` - Anexe uma imagem junto com este comando para detectar objetos
- `!ajuda` - Exibe informações de ajuda

## Como criar um Bot no Discord

1. Acesse o [Portal de Desenvolvedores do Discord](https://discord.com/developers/applications)
2. Clique em "New Application" e dê um nome para sua aplicação
3. Navegue até a seção "Bot" e clique em "Add Bot"
4. Em "Privileged Gateway Intents", ative "MESSAGE CONTENT INTENT"
5. Copie o token do bot e coloque-o no arquivo `config.env`
6. Para convidar o bot para um servidor, vá para a seção "OAuth2" > "URL Generator"
7. Selecione os escopos "bot" e "applications.commands"
8. Nas permissões do bot, selecione pelo menos:
   - "Send Messages"
   - "Attach Files"
   - "Read Message History"
9. Copie e acesse a URL gerada para adicionar o bot ao seu servidor

## Exemplo de uso

1. Digite `!detect` em qualquer canal onde o bot tenha permissões
2. Anexe uma imagem à sua mensagem
3. O bot processará a imagem e responderá com os objetos detectados 