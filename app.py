from flask import Flask, render_template, request
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import requests


#confioguracion FLASK
app = Flask(__name__)
conversation = [] #Guarda preguntas y repsuestas

#carga Moledo de embedding
embedder = SentenceTransformer('all-MiniLM-L6-V2')

#Chroma DB local
client = PersistentClient(path='./chroma')
collection = client.get_or_create_collection(name='documentos')

#modelo de CHAT
OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL = 'gpt-oss:20b'

@app.route("/", methods=['GET', 'POST'])
def index():
  global conversation
  if request.method == 'POST':
    user_input = request.form['user_input']
    
    embedding_input = embedder.encode(user_input).tolist()

    #texto similar
    resultados = collection.query(query_embeddings=[embedding_input], n_results=5)
    contexto = resultados['documents'][0][0]

    promt = f"""
            Eres un agente aduanal experto en reglamentos de liquidación aduanera y OEA. 

            ANÁLISIS DE LA PREGUNTA:
            - Si es una pregunta específica/fáctica (qué, cómo, cuándo, dónde, requisitos, plazos, montos): respuesta CONCISA
            - Si es una pregunta compleja (explicación, procedimiento, comparación, análisis): respuesta DETALLADA

            REGLAS:
            1. Basarte estrictamente en el contexto proporcionado
            2. Ser 40% creativo SOLO en ejemplos y analogías (no en información sustantiva)
            3. Mantener lenguaje profesional pero accesible
            4. Estructurar respuestas largas con encabezados claros
            5. Usar viñetas para listas en respuestas extensas

            CONTEXTO:
            \"\"\"{contexto}\"\"\"

            TIPO DE RESPUESTA REQUERIDA: (analiza el tipo de pregunta y determina concisa o extensa)

            PREGUNTA: {user_input}
            """

    payload ={
      'model': MODEL,
      'prompt': promt,
      'stream': False
    }

    response = requests.post(OLLAMA_URL, json = payload)
    result = response.json()

    conversation.append(('Tu', user_input))
    conversation.append(('IA', result['response']))
    print(conversation)

  return render_template('index.html', conversation = conversation)



if __name__ == '__main__':
  app.run(debug=True)