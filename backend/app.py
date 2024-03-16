__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3') #this is how u fix weird sqlite3 issue
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import chromadb
from ai import AI
# from langchain_community import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
CORS(app, origins="*")

# add client = chromadb.Client() to app.config so that it can be accessed by the routes
# client_object = chromadb.PersistentClient()
# app.config["client"] = client_object


ai = AI()

PERSISTENT_DIR="./data"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=PERSISTENT_DIR, embedding_function=embeddings)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    """
    Populates the collection with the given file path.

    Once the documents are fetched from IFPS/Filecoin,
    they are sent to this route to be vectorized and stored in the collection.
    """
    data = request.json
    list_of_paths = data["paths"]# list of strings
    collection_name = data["collection_name"]

    # DB STUFF
    # check if the collection exists
    collection_exists = ai.check_if_collection_exists(vector_db, collection_name)
    if not collection_exists: 
        # create the collection and put it in the database
        vector_db = ai.create_collection_and_put_it_in_db(list_of_paths, collection_name)
    else:
        # load the collection from the database
        vector_db = ai.load_collection_from_db(collection_name)

    # get the retriever for the collection
    retriever = ai.get_retriever_for_given_vectordb(vector_db)

    answer = ai.get_answer(data["question"], retriever)
    
    return jsonify({"answer": answer})


@app.route("/add_live", methods=["POST"])
def live():
    data = request.json
    audio_text = data["text"]
    first_time = data["first_time"]
    if first_time:
        video_path = agent.get_video_path(audio_text, first_time=True)
    else:
        video_path = agent.get_video_path(audio_text)
    return jsonify({"video_path": video_path})

if __name__ == '__main__':
    # app.run(port=8000, debug=True)
    app.run(host='0.0.0.0', port=8000, debug=True)

    