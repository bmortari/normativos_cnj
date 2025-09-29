# debug_single_case.py
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- CONFIGURAÇÕES ---
load_dotenv()
K_RETRIEVAL = 10
ID_PROBLEMATICO = 21543  # O ID que está causando o erro

# --- Modelos (mesmos do seu script original) ---
RERANKER_MODEL = 'unicamp-dl/mMiniLM-L6-v2-en-pt-msmarco-v2'
RETRIEVER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# --- Tabelas relevantes para o experimento que falha ---
SILVER_SET_TABLE = "public.silver_set_minilm_structured"
VECTOR_TABLE = "public.normativos_cnj_minilm_structured"

def get_db_connection():
    db_params = {
        "host": os.getenv("DB_HOST"), "port": os.getenv("DB_PORT"),
        "dbname": os.getenv("DB_NAME_POSTGRES"), "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASS")
    }
    conn = psycopg2.connect(**db_params)
    conn.autocommit = True
    return conn

def main():
    print("Iniciando script de depuração...")
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1. Carregar os modelos
        print(f"Carregando retriever: {RETRIEVER_MODEL}")
        retriever_model = SentenceTransformer(RETRIEVER_MODEL)
        print(f"Carregando re-ranker: {RERANKER_MODEL}")
        reranker = CrossEncoder(RERANKER_MODEL)

        # 2. Obter a pergunta problemática do banco de dados
        print(f"\nBuscando pergunta para id_original = {ID_PROBLEMATICO} na tabela {SILVER_SET_TABLE}")
        query_pergunta = sql.SQL("SELECT pergunta FROM {} WHERE id_original = %s").format(
            sql.Identifier(*SILVER_SET_TABLE.split('.'))
        )
        cursor.execute(query_pergunta, (ID_PROBLEMATICO,))
        result = cursor.fetchone()
        if not result:
            print("ERRO: Pergunta não encontrada no banco de dados!")
            return
        
        question = result[0]
        print(f"Pergunta encontrada. Comprimento: {len(question)} caracteres.")
        print("-" * 50)
        print(f"Texto da Pergunta: {question[:500]}...") # Mostra os primeiros 500 caracteres
        print("-" * 50)


        # 3. Simular a etapa de Retrieval
        print("\nCodificando a pergunta para o retrieval...")
        query_vector = retriever_model.encode(question)

        print(f"Buscando os {K_RETRIEVAL} documentos mais próximos na tabela {VECTOR_TABLE}")
        query_busca = sql.SQL("""
            SELECT id, document FROM {table} ORDER BY embedding <=> %s LIMIT %s;
        """).format(table=sql.Identifier(*VECTOR_TABLE.split('.')))
        
        vector_str = str(query_vector.tolist())
        cursor.execute(query_busca, (vector_str, K_RETRIEVAL))
        retrieved_docs = [{'id': item[0], 'document': item[1]} for item in cursor.fetchall()]
        
        print(f"{len(retrieved_docs)} documentos recuperados. Analisando comprimentos:")
        for i, doc in enumerate(retrieved_docs):
            doc_len = len(str(doc.get('document', '')))
            print(f"  Doc {i+1} (ID: {doc['id']}): {doc_len} caracteres")
            if doc_len > 5000:
                print(f"    AVISO: Documento {i+1} é MUITO LONGO!")


        # 4. Simular a etapa de Re-ranking (com a correção robusta)
        print("\nPreparando dados para o re-ranker (com truncamento)...")
        MAX_CHARS_QUESTION = 2000 # Truncamento seguro para a pergunta
        MAX_CHARS_DOC = 2000      # Truncamento seguro para o documento

        reranker_input = []
        safe_question = question[:MAX_CHARS_QUESTION]

        for doc in retrieved_docs:
            document_text = str(doc.get('document', ''))[:MAX_CHARS_DOC]
            reranker_input.append([safe_question, document_text])

        print("Tentando executar reranker.predict()... Se quebrar aqui, o problema é confirmado.")
        
        # Para depuração, vamos executar um por um
        for i, pair in enumerate(reranker_input):
            print(f"  Processando par {i+1}/{len(reranker_input)}...")
            try:
                # O método predict aceita um único par ou uma lista de pares
                score = reranker.predict(pair) 
                print(f"    Sucesso! Score: {score}")
            except Exception as e:
                print(f"    !!! ERRO AO PROCESSAR O PAR {i+1} !!!")
                print(f"    Pergunta (truncada): {len(pair[0])} chars")
                print(f"    Documento (truncado): {len(pair[1])} chars")
                print(f"    Erro: {e}")
                # Vamos tentar rodar o lote inteiro para replicar o erro original
                print("\nTentando replicar o erro com o lote inteiro...")
                reranker.predict(reranker_input)

        print("\nExecutando predição em lote (como no script original)...")
        scores = reranker.predict(reranker_input)
        
        print("\n--- SUCESSO! ---")
        print("O script de depuração executou sem erros com os textos truncados.")

    except Exception as e:
        print(f"\n--- FALHA ---")
        print(f"O script de depuração falhou com o erro: {e}")
    finally:
        cursor.close()
        conn.close()
        print("\nConexão com o banco de dados fechada.")

if __name__ == "__main__":
    main()