import os
import io
import discord
from discord.ext import commands
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from collections import Counter
from datetime import datetime
import asyncio
import aiohttp
import threading

# Adicionar classes seguras para o PyTorch 2.6
try:
    from ultralytics.nn.tasks import DetectionModel
    import torch.serialization
    torch.serialization.add_safe_globals([DetectionModel])
except Exception as e:
    print(f"Aviso: Não foi possível adicionar classes seguras: {str(e)}")

# Carregar variáveis de ambiente
try:
    load_dotenv('config.env')
except:
    print("Arquivo config.env não encontrado. Certifique-se de criar o arquivo com base no config.env.example")

# Configuração do bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Configuração global
CONFIG = {
    'model_size': 'n',  # n=nano, s=small, m=medium, l=large, x=xlarge
    'confidence_threshold': 0.5,
    'max_objects': 20,
    'color_analysis': True
}

# Dicionário para rastrear downloads em andamento
DOWNLOADS_EM_ANDAMENTO = {}

# Função para baixar modelos em uma thread separada
def download_model_threaded(url, path, tamanho, message_id, channel_id):
    try:
        print(f"Iniciando download do modelo YOLOv8{tamanho} em thread separada...")
        import urllib.request
        
        # Função para reportar progresso
        def report_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            if percent % 10 == 0:  # Reportar a cada 10%
                print(f"Download do modelo YOLOv8{tamanho}: {percent}% concluído")
                # Não podemos atualizar o Discord diretamente de uma thread,
                # então apenas armazenamos o progresso para consultas posteriores
                DOWNLOADS_EM_ANDAMENTO[tamanho] = {
                    'percent': percent,
                    'status': 'downloading'
                }
        
        # Fazer o download com relatório de progresso
        urllib.request.urlretrieve(url, path, reporthook=report_progress)
        
        print(f"Download do modelo YOLOv8{tamanho} concluído!")
        DOWNLOADS_EM_ANDAMENTO[tamanho] = {
            'percent': 100,
            'status': 'completed'
        }
    except Exception as e:
        print(f"Erro no download do modelo: {str(e)}")
        DOWNLOADS_EM_ANDAMENTO[tamanho] = {
            'percent': 0,
            'status': 'error',
            'error': str(e)
        }

# Carregar modelo YOLO
async def load_yolo_model(size='n', ctx=None):
    print(f"Carregando modelo YOLO de tamanho '{size}'...")
    model_path = f'yolov8{size}.pt'
    
    if not os.path.exists(model_path):
        if ctx:
            await ctx.send(f"Modelo YOLOv8{size} não encontrado. Iniciando download... (isso pode levar vários minutos para modelos maiores)")
        print(f"Baixando o modelo YOLOv8{size}...")
        
        # URL do modelo
        url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8{size}.pt"
        
        # Verificar o tamanho do modelo antes de baixar
        modelo_sizes = {
            'n': "6 MB",
            's': "22 MB",
            'm': "50 MB",
            'l': "90 MB",
            'x': "670 MB"
        }
        
        if size in modelo_sizes and ctx:
            await ctx.send(f"⬇️ Baixando YOLOv8{size} ({modelo_sizes[size]})...")
        
        # Para modelos maiores (m, l, x), usar download em thread separada
        if size in ['m', 'l', 'x']:
            # Armazenar informações de mensagem para atualizações
            message_id = ctx.message.id if ctx else None
            channel_id = ctx.channel.id if ctx else None
            
            # Iniciar download em uma thread separada
            DOWNLOADS_EM_ANDAMENTO[size] = {
                'percent': 0,
                'status': 'starting'
            }
            
            download_thread = threading.Thread(
                target=download_model_threaded,
                args=(url, model_path, size, message_id, channel_id)
            )
            download_thread.start()
            
            # Se temos um contexto, reportar progresso periodicamente
            if ctx:
                progress_message = await ctx.send("Progresso: 0%")
                
                while DOWNLOADS_EM_ANDAMENTO[size]['status'] == 'starting' or DOWNLOADS_EM_ANDAMENTO[size]['status'] == 'downloading':
                    if DOWNLOADS_EM_ANDAMENTO[size]['status'] == 'downloading':
                        percent = DOWNLOADS_EM_ANDAMENTO[size]['percent']
                        await progress_message.edit(content=f"Progresso: {percent}% | {'▓' * (percent // 10)}{'░' * (10 - percent // 10)}")
                    await asyncio.sleep(5)  # Verificar a cada 5 segundos
                
                status = DOWNLOADS_EM_ANDAMENTO[size]['status']
                if status == 'completed':
                    await progress_message.edit(content=f"✅ Download concluído! Carregando modelo...")
                elif status == 'error':
                    error = DOWNLOADS_EM_ANDAMENTO[size].get('error', 'desconhecido')
                    await progress_message.edit(content=f"❌ Erro no download: {error}")
                    raise Exception(f"Erro no download do modelo: {error}")
            else:
                # Se não temos contexto, apenas esperar o download terminar
                download_thread.join()
                status = DOWNLOADS_EM_ANDAMENTO[size]['status']
                if status != 'completed':
                    error = DOWNLOADS_EM_ANDAMENTO[size].get('error', 'desconhecido')
                    raise Exception(f"Erro no download do modelo: {error}")
        else:
            # Para modelos menores, baixar diretamente
            try:
                import urllib.request
                urllib.request.urlretrieve(url, model_path)
                print("Modelo baixado com sucesso!")
                if ctx:
                    await ctx.send("✅ Download concluído! Carregando modelo...")
            except Exception as e:
                print(f"Erro ao baixar modelo: {str(e)}")
                if ctx:
                    await ctx.send(f"❌ Erro ao baixar modelo: {str(e)}")
                raise e
    
    # Carregar modelo com opção weights_only=False
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print(f"Modelo YOLOv8{size} carregado com sucesso!")
        return model
    except Exception as e1:
        print(f"Erro ao carregar o modelo normalmente: {str(e1)}")
        print("Tentando com método alternativo...")
        
        # Método alternativo forçando weights_only=False
        try:
            import torch
            orig_load = torch.load
            def patched_load(f, *args, **kwargs):
                kwargs['weights_only'] = False
                return orig_load(f, *args, **kwargs)
            
            # Aplicar patch temporário
            torch.load = patched_load
            from ultralytics import YOLO
            model = YOLO(model_path)
            # Restaurar função original
            torch.load = orig_load
            print(f"Modelo YOLOv8{size} carregado com sucesso usando método alternativo!")
            return model
        except Exception as e2:
            print(f"Erro no método alternativo: {str(e2)}")
            print("Por favor, verifique a instalação do PyTorch e Ultralytics")
            if ctx:
                await ctx.send(f"❌ Erro ao carregar o modelo: {str(e2)}")
            raise e2

# Carregar modelo inicial em background após o bot iniciar
model = None  # Será carregado após o bot iniciar

# Armazenar hora de início
@bot.event
async def on_ready():
    global model
    import time
    bot.start_time = time.time()
    
    print(f'{bot.user.name} está online!')
    print(f'ID do Bot: {bot.user.id}')
    print('------')
    
    # Carregar modelo inicial em background
    try:
        model = await load_yolo_model(CONFIG['model_size'])
        print(f"Modelo inicial YOLOv8{CONFIG['model_size']} carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelo inicial: {str(e)}")

@bot.event
async def on_message(message):
    # Ignorar mensagens do próprio bot
    if message.author == bot.user:
        return
    
    # Verificar se a mensagem começa com 'detect' sem o prefixo
    if message.content.lower().startswith('detect') and message.attachments:
        print(f"Comando 'detect' sem prefixo detectado de {message.author.name}")
        ctx = await bot.get_context(message)
        await detect(ctx)
    # Verificar se a mensagem é apenas '!detect' sem anexos, mas há anexos na mensagem
    elif message.content.lower() in ['!detect', 'detect'] and message.attachments:
        print(f"Comando detect com anexos detectado de {message.author.name}")
        ctx = await bot.get_context(message)
        await detect(ctx)
    else:
        # Processamento normal de comandos
        await bot.process_commands(message)

@bot.command()
async def modelo(ctx, tamanho=None):
    """Altera o tamanho do modelo YOLO (n=nano, s=small, m=medium, l=large, x=xlarge)"""
    global model, CONFIG
    
    tamanhos_validos = ['n', 's', 'm', 'l', 'x']
    
    if not tamanho:
        await ctx.send(f"**Tamanho atual do modelo:** YOLOv8{CONFIG['model_size']}\n" + 
                       f"Tamanhos disponíveis: {', '.join(['n (nano)', 's (small)', 'm (medium)', 'l (large)', 'x (xlarge)'])}\n" +
                       "Use `!modelo <tamanho>` para mudar. Exemplo: `!modelo s`")
        return
    
    if tamanho not in tamanhos_validos:
        await ctx.send(f"Tamanho inválido! Use um dos seguintes: {', '.join(tamanhos_validos)}")
        return
    
    if tamanho in ['l', 'x']:
        await ctx.send(f"⚠️ Aviso: O modelo YOLOv8{tamanho} é muito grande e pode levar vários minutos para baixar. Tem certeza que deseja continuar? Digite `!confirmar modelo {tamanho}` para confirmar.")
        return
    
    await ctx.send(f"Carregando modelo YOLOv8{tamanho}... Isso pode levar alguns segundos.")
    
    try:
        new_model = await load_yolo_model(tamanho, ctx)
        model = new_model
        CONFIG['model_size'] = tamanho
        await ctx.send(f"✅ Modelo YOLOv8{tamanho} carregado com sucesso!")
    except Exception as e:
        await ctx.send(f"❌ Erro ao carregar o modelo: {str(e)}")

@bot.command()
async def confirmar(ctx, tipo=None, tamanho=None):
    """Confirma operações potencialmente demoradas"""
    if tipo == 'modelo' and tamanho:
        await ctx.send(f"Confirmado! Carregando modelo YOLOv8{tamanho}... Este processo pode levar vários minutos.")
        try:
            new_model = await load_yolo_model(tamanho, ctx)
            global model, CONFIG
            model = new_model
            CONFIG['model_size'] = tamanho
            await ctx.send(f"✅ Modelo YOLOv8{tamanho} carregado com sucesso!")
        except Exception as e:
            await ctx.send(f"❌ Erro ao carregar o modelo: {str(e)}")
    else:
        await ctx.send("Comando inválido. Use `!confirmar modelo <tamanho>` para confirmar a alteração do modelo.")

@bot.command()
async def config(ctx, param=None, valor=None):
    """Altera configurações do bot de detecção"""
    global CONFIG
    
    if not param:
        # Mostrar configurações atuais
        config_msg = "**Configurações Atuais:**\n"
        for key, value in CONFIG.items():
            config_msg += f"- **{key}**: {value}\n"
        config_msg += "\nPara alterar: `!config <parametro> <valor>`"
        await ctx.send(config_msg)
        return
    
    # Verificar se o parâmetro existe
    if param not in CONFIG:
        await ctx.send(f"Parâmetro desconhecido: {param}. Parâmetros disponíveis: {', '.join(CONFIG.keys())}")
        return
    
    if not valor:
        await ctx.send(f"Valor atual de **{param}**: {CONFIG[param]}")
        return
    
    # Converter o valor para o tipo adequado
    try:
        if param == 'confidence_threshold':
            valor = float(valor)
            if not (0 <= valor <= 1):
                await ctx.send("Valor de confiança deve estar entre 0 e 1")
                return
        elif param == 'max_objects':
            valor = int(valor)
            if valor < 1:
                await ctx.send("Número máximo de objetos deve ser pelo menos 1")
                return
        elif param == 'color_analysis':
            valor = valor.lower() in ['true', 'yes', 'sim', '1', 'on', 'ativado']
        
        # Atualizar configuração
        CONFIG[param] = valor
        await ctx.send(f"✅ **{param}** atualizado para: **{valor}**")
    except ValueError:
        await ctx.send(f"Valor inválido para {param}. Verifique o tipo de dado.")

# Função para analisar cores predominantes na imagem
def analyze_colors(image, num_colors=5):
    # Redimensionar imagem para processamento mais rápido
    img_small = image.copy()
    img_small.thumbnail((100, 100))
    
    # Converter para RGB se necessário
    if img_small.mode != 'RGB':
        img_small = img_small.convert('RGB')
    
    # Obter pixels
    pixels = np.array(img_small)
    pixels = pixels.reshape(-1, 3)
    
    # Agrupar cores similares
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
    
    # Simplificar cores (reduzir precisão para agrupar cores similares)
    pixels_simple = (pixels // 32) * 32
    pixels_rgb = [tuple(pixel) for pixel in pixels_simple]
    
    # Contar cores
    color_counts = Counter(pixels_rgb)
    
    # Obter as cores mais comuns
    most_common = color_counts.most_common(num_colors)
    total_pixels = len(pixels_rgb)
    
    # Formatar resultados
    colors = []
    for color, count in most_common:
        percentage = count / total_pixels * 100
        hex_color = rgb_to_hex(color)
        colors.append({
            'rgb': color,
            'hex': hex_color,
            'percentage': percentage
        })
    
    return colors

@bot.command()
async def detect(ctx):
    """Comando para detectar objetos em uma imagem utilizando YOLO"""
    print(f"Comando detect recebido de {ctx.author.name}")
    
    if not ctx.message.attachments:
        print("Nenhum anexo encontrado na mensagem")
        await ctx.send("Por favor, anexe uma imagem junto com o comando.")
        return
    
    attachment = ctx.message.attachments[0]
    print(f"Anexo encontrado: {attachment.filename}")
    
    # Verificar se é uma imagem
    valid_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']
    if not any(attachment.filename.lower().endswith(ext) for ext in valid_extensions):
        print(f"Arquivo inválido: {attachment.filename}")
        await ctx.send(f"Por favor, anexe um arquivo de imagem válido. Formatos suportados: {', '.join(valid_extensions)}")
        return
    
    print("Baixando imagem...")
    # Baixar a imagem
    image_bytes = await attachment.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Converter para RGB se a imagem for RGBA para evitar problemas ao salvar como JPEG
    if image.mode == 'RGBA':
        print("Convertendo imagem RGBA para RGB...")
        image = image.convert('RGB')
    
    # Salvar temporariamente a imagem
    temp_img_path = "temp_image.jpg"
    image.save(temp_img_path)
    print(f"Imagem salva temporariamente em: {temp_img_path}")
    
    # Analisar cores predominantes se configurado
    color_info = ""
    if CONFIG['color_analysis']:
        try:
            print("Analisando cores predominantes...")
            colors = analyze_colors(image)
            color_info = "\n**Cores Predominantes:**\n"
            for i, color in enumerate(colors):
                color_info += f"{i+1}. {color['hex']} ({color['percentage']:.1f}%)\n"
        except Exception as ce:
            print(f"Erro na análise de cores: {str(ce)}")
    
    # Mensagem enquanto processa
    await ctx.send("Processando imagem... Aguarde um momento.")
    
    try:
        # Realizar a detecção
        print("Iniciando detecção com YOLO...")
        try:
            # Usar threshold de confiança da configuração
            results = model(temp_img_path, conf=CONFIG['confidence_threshold'])
            print("Detecção concluída. Processando resultados...")
        except Exception as yolo_error:
            print(f"ERRO na detecção YOLO: {str(yolo_error)}")
            await ctx.send(f"Erro ao processar a imagem com YOLO: {str(yolo_error)}")
            # Limpar arquivo temporário em caso de erro
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            return
        
        # Processar resultados
        result = results[0]
        result_img = result.plot()
        result_image = Image.fromarray(result_img)
        
        # Adicionar informações na imagem
        draw = ImageDraw.Draw(result_image)
        try:
            # Tenta carregar uma fonte
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            # Se não conseguir, usa fonte padrão
            font = ImageFont.load_default()
        
        # Adicionar marca d'água
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        draw.text((10, 10), f"YOLOv8{CONFIG['model_size']} - {timestamp}", fill=(255, 255, 255), font=font)
        
        # Salvar a imagem com as detecções
        output_path = "detection_result.jpg"
        result_image.save(output_path)
        print(f"Imagem com detecções salva em: {output_path}")
        
        # Contar objetos detectados
        boxes = result.boxes
        class_counts = {}
        class_confidences = {}
        detected_objects = []
        
        print("Objetos detectados:")
        for box in boxes:
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            confidence = float(box.conf[0].item())
            
            # Extrair coordenadas da bounding box (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Calcular posição relativa na imagem
            img_width, img_height = image.size
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            position_x = center_x / img_width
            position_y = center_y / img_height
            
            # Determinar posição em termos de região
            region = ""
            if position_y < 0.33:
                region += "superior "
            elif position_y > 0.66:
                region += "inferior "
            else:
                region += "centro "
                
            if position_x < 0.33:
                region += "esquerdo"
            elif position_x > 0.66:
                region += "direito"
            else:
                region += "central"
            
            # Armazenar dados do objeto detectado
            detected_objects.append({
                "class": class_name,
                "confidence": confidence,
                "width": width,
                "height": height,
                "area": area,
                "region": region
            })
            
            print(f"- {class_name}: {confidence:.2f} (tamanho: {width:.1f}x{height:.1f})")
            
            # Atualizar contagens
            if class_name in class_counts:
                class_counts[class_name] += 1
                class_confidences[class_name].append(confidence)
            else:
                class_counts[class_name] = 1
                class_confidences[class_name] = [confidence]
        
        # Ordenar objetos por tamanho
        detected_objects.sort(key=lambda x: x["area"], reverse=True)
        
        # Criar mensagem de detecção
        detection_message = f"**Análise com YOLOv8{CONFIG['model_size']} (conf: {CONFIG['confidence_threshold']}):**\n\n"
        detection_message += "**Objetos Detectados:**\n"
        
        # Adicionar resumo por classe
        for class_name, count in class_counts.items():
            avg_confidence = sum(class_confidences[class_name]) / len(class_confidences[class_name])
            detection_message += f"- {class_name}: {count} (confiança média: {avg_confidence:.2%})\n"
        
        # Adicionar detalhes de cada objeto (limitado pelo max_objects da configuração)
        if detected_objects:
            detection_message += "\n**Detalhes dos Objetos:**\n"
            for i, obj in enumerate(detected_objects[:CONFIG['max_objects']]):
                detection_message += f"{i+1}. {obj['class']}: confiança {obj['confidence']:.2%}, tamanho {obj['width']:.1f}x{obj['height']:.1f} pixels, {obj['region']}\n"
        
        # Adicionar estatísticas gerais
        detection_message += "\n**Estatísticas:**\n"
        detection_message += f"- Total de objetos: {len(detected_objects)}\n"
        detection_message += f"- Classes detectadas: {len(class_counts)}\n"
        if detected_objects:
            max_obj = max(detected_objects, key=lambda x: x["confidence"])
            detection_message += f"- Objeto com maior confiança: {max_obj['class']} ({max_obj['confidence']:.2%})\n"
            largest_obj = max(detected_objects, key=lambda x: x["area"])
            detection_message += f"- Maior objeto: {largest_obj['class']} ({largest_obj['width']:.1f}x{largest_obj['height']:.1f} pixels)\n"
        
        # Adicionar análise de cores
        detection_message += color_info
        
        # Adicionar timestamp
        detection_message += f"\n*Processado em: {timestamp}*"
        
        # Enviar resultado
        print("Enviando resultado para o Discord...")
        await ctx.send(detection_message, file=discord.File(output_path))
        print("Resultado enviado com sucesso!")
        
        # Limpar arquivos temporários
        os.remove(temp_img_path)
        os.remove(output_path)
        print("Arquivos temporários removidos")
        
    except Exception as e:
        print(f"ERRO na detecção: {str(e)}")
        await ctx.send(f"Ocorreu um erro ao processar a imagem: {str(e)}")
        # Limpar arquivo temporário em caso de erro
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

@bot.command()
async def ajuda(ctx):
    """Exibe informações de ajuda sobre o bot"""
    help_text = """
**Bot de Detecção YOLO v2.0**

**Comandos disponíveis:**
`!detect` - Anexe uma imagem com este comando para detectar objetos nela
`!modelo [tamanho]` - Verifica ou altera o tamanho do modelo YOLO (n, s, m, l, x)
`!config [param] [valor]` - Verifica ou altera configurações de detecção
`!ajuda` - Exibe esta mensagem de ajuda

**Exemplos de uso:**
1. `!detect` - Anexe uma imagem para detectar objetos
2. `!modelo s` - Muda para o modelo small (mais preciso, mais lento)
3. `!config confidence_threshold 0.3` - Reduz o limite de confiança para 0.3 (30%)
4. `!config color_analysis false` - Desativa a análise de cores

**Tamanhos de modelo disponíveis:**
- `n` (nano): Mais rápido, menos preciso
- `s` (small): Bom equilíbrio entre velocidade e precisão
- `m` (medium): Médio porte, mais preciso
- `l` (large): Grande, alta precisão
- `x` (xlarge): Muito grande, precisão máxima, mais lento

**Configurações disponíveis:**
- `confidence_threshold`: Limite de confiança (0.0-1.0)
- `max_objects`: Número máximo de objetos a mostrar nos detalhes
- `color_analysis`: Análise de cores predominantes (true/false)
    """
    await ctx.send(help_text)

@bot.command()
async def status(ctx):
    """Mostra o status atual do bot e downloads em andamento"""
    global DOWNLOADS_EM_ANDAMENTO, CONFIG
    
    status_msg = "**Status do Bot de Detecção YOLO**\n\n"
    
    # Informações do modelo
    status_msg += f"**Modelo Atual:** YOLOv8{CONFIG['model_size']}\n"
    status_msg += f"**Configurações:**\n"
    for key, value in CONFIG.items():
        status_msg += f"- {key}: {value}\n"
    
    # Downloads em andamento
    if DOWNLOADS_EM_ANDAMENTO:
        status_msg += "\n**Downloads em Andamento:**\n"
        for size, info in DOWNLOADS_EM_ANDAMENTO.items():
            status = info['status']
            if status == 'downloading':
                percent = info['percent']
                status_msg += f"- YOLOv8{size}: {percent}% | {'▓' * (percent // 10)}{'░' * (10 - percent // 10)}\n"
            elif status == 'completed':
                status_msg += f"- YOLOv8{size}: ✅ Concluído\n"
            elif status == 'error':
                error = info.get('error', 'desconhecido')
                status_msg += f"- YOLOv8{size}: ❌ Erro ({error})\n"
            else:
                status_msg += f"- YOLOv8{size}: {status}\n"
    
    # Informações do sistema
    status_msg += "\n**Informações do Sistema:**\n"
    try:
        import psutil
        # Uso de memória
        mem = psutil.virtual_memory()
        status_msg += f"- Memória: {mem.percent}% usado ({mem.used / (1024**3):.1f}GB / {mem.total / (1024**3):.1f}GB)\n"
        # Uso de CPU
        status_msg += f"- CPU: {psutil.cpu_percent()}% usado\n"
    except ImportError:
        status_msg += "- Informações do sistema não disponíveis (psutil não instalado)\n"
    
    # Tempo online
    import time
    try:
        uptime = time.time() - bot.start_time
        hours, rem = divmod(uptime, 3600)
        minutes, seconds = divmod(rem, 60)
        status_msg += f"- Tempo online: {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
    except:
        pass
    
    await ctx.send(status_msg)

# Iniciar o bot
if __name__ == "__main__":
    token = os.getenv('DISCORD_TOKEN')
    if token:
        bot.run(token)
    else:
        print("Token do Discord não encontrado. Verifique seu arquivo config.env") 