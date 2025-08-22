import PyPDF2
from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

#Chroma DB local
client = PersistentClient(path='./chroma')
collection = client.get_or_create_collection(name='documentos')

#carga Moledo de embedding
embedder = SentenceTransformer('all-MiniLM-L6-V2')

def carga_text_embeding():
  texto = extraer_text('./Documents/circular0382025 (2).pdf')
  fracmentos = dividir_texto(texto)

  embeddings = embedder.encode(fracmentos).tolist()

  for i, fracmento in enumerate(fracmentos):
    collection.add(documents=[fracmento], ids=[f'frag {i}'], embeddings=[embeddings[i]])

  print (len(fracmentos), len(embeddings))

def extraer_text(ruta):
  texto = ""
  try:
    with open(ruta, 'rb') as archivo:
      # Crear objeto PDF Reader
      pdf_reader = PyPDF2.PdfReader(archivo)
            
      # Obtener número de páginas
      num_paginas = len(pdf_reader.pages)
      print(f"El PDF tiene {num_paginas} páginas")
            
      # Extraer texto de cada página
      for pagina_num, pagina in enumerate(pdf_reader.pages):
        texto_pagina = pagina.extract_text()
        if texto_pagina.strip():  # Solo si hay texto
          texto += f"--- Página {pagina_num + 1} ---\n"
          texto += texto_pagina + "\n\n"
                
      return texto
    
  except Exception as e:
    print(f"Error al leer el PDF: {e}")
    return None
  

def dividir_texto(texto: str, largo: int = 1000, superpos: int = 100) -> list:
  if not texto:
    return []
    
  fragmentos = []
  inicio = 0
  texto_len = len(texto)
    
  while inicio < texto_len:
    if inicio == 0:
      fragmentos.append(texto[:largo])
      inicio = largo - superpos
    else:
      fragmentos.append(texto[inicio : inicio + largo])
      inicio = inicio + largo - superpos
        
  return fragmentos
  
carga_text_embeding()