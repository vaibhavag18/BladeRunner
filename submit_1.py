from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from transformers import LLaMATokenizer, LLaMAForCausalLM
from sentence-transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import numpy as np

# Define your Cassandra credentials
ASTRA_DB_SECURE_BUNDLE_PATH = "path_to_secure_bundle.zip"
ASTRA_DB_APPLICATION_TOKEN = "your_application_token"
ASTRA_DB_CLIENT_ID = "your_client_id"
ASTRA_DB_CLIENT_SECRET = "your_client_secret"
ASTRA_DB_KEYSPACE = "your_keyspace"

# Define the Hugging Face API key
hf_api_key = "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"

# Define the model name and load tokenizer and model
model_name = "huggingface/llama"
tokenizer = LLaMATokenizer.from_pretrained(model_name, use_auth_token=hf_api_key)
model = LLaMAForCausalLM.from_pretrained(model_name, use_auth_token=hf_api_key)

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return ' '.join(text)

# Function to preprocess text into chunks
def preprocess_text(text, chunk_size=1000):
    text = text.replace('\n', ' ')
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Connect to Cassandra cluster
cloud_config = {'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH}
auth_provider = PlainTextAuthProvider(ASTRA_DB_CLIENT_ID, ASTRA_DB_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect(ASTRA_DB_KEYSPACE)

# Load the SentenceTransformer model for embeddings
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# Function to insert embeddings and text chunks into Cassandra
def insert_embeddings(text_chunks, text_embeddings):
    query = "INSERT INTO your_table (chunk_id, chunk_text, chunk_embedding) VALUES (?, ?, ?)"
    for i, (chunk, embedding) in enumerate(zip(text_chunks, text_embeddings)):
        embedding_bytes = np.array(embedding).tobytes()
        session.execute(query, (i, chunk, embedding_bytes))

# Function to query embeddings from Cassandra
def query_embeddings(query):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    select_query = "SELECT chunk_id, chunk_text, chunk_embedding FROM your_table"
    rows = session.execute(select_query)

    best_similarity = -1
    best_match_text = None

    for row in rows:
        chunk_id = row.chunk_id
        chunk_text = row.chunk_text
        chunk_embedding_bytes = row.chunk_embedding

        # Convert bytes to numpy array
        chunk_embedding = np.frombuffer(chunk_embedding_bytes, dtype=np.float32)

        # Calculate similarity
        similarity = util.pytorch_cos_sim(query_embedding, chunk_embedding)[0]

        # Check if this is the best match
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_text = chunk_text

    return best_match_text

# Load the PDF and preprocess the text
pdf_path = 'blade_runner_2049.pdf'
pdf_text = extract_text_from_pdf(pdf_path)
text_chunks = preprocess_text(pdf_text)

# Generate embeddings and insert them into Cassandra
text_embeddings = embedding_model.encode(text_chunks, convert_to_tensor=True)
insert_embeddings(text_chunks, text_embeddings)

# List of sample questions
sample_questions = [
    "Explain the theme of the movie?",
    "Who are the characters?",
    "How many male and female characters are in the movie?",
    "Does the script pass the Bechdel test?",
    "What is the role of Deckard in the movie?",
]

# Main loop for taking user queries and providing answers
while True:
    query = input("\nEnter your question (or type 'quit' to exit):\n")

    # Provide sample questions for guidance
    if query.lower() == 'help':
        print("\nHere are some sample questions you can ask:")
        for sample in sample_questions:
            print(f" - {sample}")
        continue

    # Break the loop if user chooses to quit
    if query.lower() == 'quit':
        break

    # Process the query
    relevant_text_chunk = query_embeddings(query)

    # Use LLaMA model to generate an answer based on the relevant text chunk
    input_ids = tokenizer(relevant_text_chunk + " " + query, return_tensors='pt').input_ids
    output = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\nAnswer: ", answer)
