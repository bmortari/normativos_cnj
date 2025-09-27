import pandas as pd
import requests
import json
from tqdm import tqdm
import time
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
import os

# --- CONFIGURAÇÕES ---
# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# URL do seu Webhook no N8N
N8N_ENDPOINT_URL = "http://157.173.125.173:5678/webhook/345e055f-c1f5-495f-aa45-db9a30e4d5d5" 

# LISTA DE TABELAS PARA PROCESSAR
# Cada dicionário contém a tabela de origem e a tabela de destino para o silver set
TABLES_TO_PROCESS = [
    {
        "source": "normativos_cnj_minilm_structured",
        "output": "silver_set_minilm_structured"
    },
    {
        "source": "normativos_cnj_minilm_fixed",
        "output": "silver_set_minilm_fixed"
    },
    {
        "source": "normativos_cnj_labse_structured",
        "output": "silver_set_labse_structured"
    },
    {
        "source": "normativos_cnj_labse_fixed",
        "output": "silver_set_labse_fixed"
    }
]

# Quantidade de textos para processar POR TABELA
SAMPLE_SIZE = 100

API_GEMINI = os.getenv("API_GEMINI")

def get_db_connection():
    """Cria e retorna uma nova conexão com o banco de dados."""
    db_params = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "dbname": os.getenv("DB_NAME_POSTGRES"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASS")
    }
    if not all(db_params.values()):
        raise ValueError("Uma ou mais variáveis de ambiente do banco de dados não foram definidas.")
    
    return psycopg2.connect(**db_params)


def carregar_dados_do_postgres(source_table_name):
    """Conecta ao PostgreSQL e carrega uma amostra de dados de uma tabela específica."""
    conn = None
    try:
        conn = get_db_connection()
        print(f"Conectado ao banco de dados para carregar dados de '{source_table_name}'.")

        # Query com o novo critério WHERE
        query_object = sql.SQL("""
            SELECT id, document 
            FROM {table_name} AS ncms
            WHERE ncms.metadata ->> 'tipo_chunk' != 'Artigo'
            ORDER BY RANDOM() 
            LIMIT %s;
        """).format(
            table_name=sql.Identifier(*source_table_name.split('.'))
        )
        
        # Converte o objeto de query para uma string segura usando o contexto da conexão
        query_string = query_object.as_string(conn)

        # Passa a string renderizada para o pandas
        df = pd.read_sql_query(query_string, conn, params=(SAMPLE_SIZE,))
        
        if df.empty:
            print("Aviso: Nenhum documento encontrado com os critérios especificados.")
        else:
            print(f"{len(df)} documentos carregados com sucesso.")
        
        return df
    except Exception as e:
        print(f"ERRO ao carregar dados do PostgreSQL da tabela '{source_table_name}': {e}")
        return None
    finally:
        if conn:
            conn.close()

def salvar_dados_no_postgres(dados, output_table_name):
    """Salva os dados gerados em uma nova tabela específica no PostgreSQL."""
    if not dados:
        print("Nenhum dado para salvar.")
        return

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        print(f"Conectado ao banco de dados para salvar dados na tabela '{output_table_name}'.")
        
        # 1. Cria a tabela se ela não existir
        create_table_query = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
                id SERIAL PRIMARY KEY,
                id_original TEXT NOT NULL,
                texto_original TEXT,
                parafrase TEXT,
                pergunta TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """).format(sql.Identifier(*output_table_name.split('.')))
        cursor.execute(create_table_query)
        print(f"Tabela '{output_table_name}' verificada/criada com sucesso.")

        # 2. Prepara os dados e a query de inserção
        insert_query = sql.SQL("""
            INSERT INTO {} (id_original, texto_original, parafrase, pergunta)
            VALUES (%s, %s, %s, %s);
        """).format(sql.Identifier(*output_table_name.split('.')))
        
        # Converte a lista de dicionários para uma lista de tuplas
        dados_para_inserir = [
            (
                item['id_resposta_esperada'], 
                item['texto_original'], 
                item['texto_parafraseado'], 
                item['pergunta_gerada']
            ) for item in dados
        ]
        
        # 3. Insere os dados em lote (muito mais eficiente)
        cursor.executemany(insert_query, dados_para_inserir)
        conn.commit()
        
        print(f"{cursor.rowcount} registros inseridos com sucesso na tabela '{output_table_name}'.")

    except Exception as e:
        print(f"ERRO ao salvar dados no PostgreSQL na tabela '{output_table_name}': {e}")
        if conn:
            conn.rollback() # Desfaz a transação em caso de erro
    finally:
        if conn:
            conn.close()
            print("Conexão para salvar dados fechada.")


def gerar_par_pergunta_resposta(doc_id, doc_text, api_gemini):
    # (Esta função permanece inalterada)
    payload = {"id": str(doc_id), "text": doc_text, "api_gemini": api_gemini}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(N8N_ENDPOINT_URL, json=payload, headers=headers, timeout=90)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"\nERRO de conexão/HTTP para o ID {doc_id}: {e}")
    return None

def processar_resposta_ia(api_response, doc_id, doc_text):
    # (Esta função permanece inalterada)
    try:
        text_str = api_response['content']['parts'][0]['text']
        start_index = text_str.find('{')
        end_index = text_str.rfind('}')
        if start_index == -1 or end_index == -1: return None
        json_str = text_str[start_index : end_index + 1]
        dados_gerados = json.loads(json_str)
        return {
            "id_resposta_esperada": str(doc_id),
            "texto_original": doc_text,
            "texto_parafraseado": dados_gerados['parafrase'],
            "pergunta_gerada": dados_gerados['pergunta']
        }
    except (KeyError, IndexError, json.JSONDecodeError, TypeError) as e:
        print(f"\nERRO ao processar a resposta da IA para o ID {doc_id}: {type(e).__name__} - {e}")
        return None

def main():
    """Função principal para orquestrar a criação dos silver sets, iterando sobre as tabelas."""
    if "webhook/345e055f-c1f5-495f-aa45-db9a30e4d5d5" not in N8N_ENDPOINT_URL:
        print("!!! ATENÇÃO: Verifique se a variável N8N_ENDPOINT_URL está configurada corretamente. !!!")
        # Removido o return para não interromper a execução caso o endpoint esteja correto.

    # Loop principal que itera sobre cada par de tabelas (origem e destino)
    for table_info in TABLES_TO_PROCESS:
        source_table = table_info["source"]
        output_table = table_info["output"]

        print("\n" + "="*80)
        print(f"INICIANDO PROCESSAMENTO PARA A TABELA: '{source_table}'")
        print(f"SALVANDO RESULTADOS EM: '{output_table}'")
        print("="*80 + "\n")

        df_amostra = carregar_dados_do_postgres(source_table)
        
        if df_amostra is None or df_amostra.empty:
            print(f"Nenhum dado foi carregado da tabela '{source_table}'. Pulando para a próxima.")
            continue
        
        silver_set_data = []
        print(f"\nIniciando a geração de dados para {len(df_amostra)} documentos de '{source_table}'...")

        for _, row in tqdm(df_amostra.iterrows(), total=len(df_amostra), desc=f"Processando '{source_table}'"):
            doc_id, doc_text = row['id'], row['document']
            time.sleep(6) # Pausa para não sobrecarregar a API
            
            api_response = gerar_par_pergunta_resposta(doc_id, doc_text, API_GEMINI)
            
            if api_response:
                processed_entry = processar_resposta_ia(api_response, doc_id, doc_text)
                if processed_entry:
                    silver_set_data.append(processed_entry)

        # Ao final do loop para uma tabela, salva todos os resultados gerados no banco
        if silver_set_data:
            print(f"\nProcessamento de '{source_table}' concluído. {len(silver_set_data)} pares gerados.")
            salvar_dados_no_postgres(silver_set_data, output_table)
        else:
            print(f"\nNenhum par pergunta/resposta foi gerado para '{source_table}'. Nada foi salvo no banco.")
            
    print("\n" + "="*80)
    print("TODAS AS TABELAS FORAM PROCESSADAS. SCRIPT FINALIZADO.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()