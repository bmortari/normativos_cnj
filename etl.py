# etl_final_otimizado.py

#!/usr/bin/env python3
"""
Script OTIMIZADO e ROBUSTO para Web Scraping, Processamento e Vetorização 
de Atos Normativos do CNJ com os modelos MiniLM e LaBSE.
"""

# --- Bibliotecas Padrão do Python ---
import random
import time
import re
import unicodedata
from urllib.parse import urljoin
import os
import json
from typing import List, Dict, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Bibliotecas de Terceiros ---
import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from tqdm import tqdm
import torch

# ==============================================================================
# PASSO 1: CONFIGURAÇÕES GERAIS E DE EXPERIMENTOS
# ==============================================================================

# --- Configurações do Scraping ---
ROOT_URL = 'https://atos.cnj.jus.br'
LISTING_URL = ROOT_URL + '/atos?atos=sim&page={}'
PAGE_LIMIT = 0      # Limite de páginas. 0 para processar todas.
LINK_LIMIT = 0     # Limite total de links. 0 para processar todos.
DELAY_ENABLED = False
DELAY_RANGE = (0.1, 0.3)

# --- Configurações de Otimização ---
MAX_WORKERS = 10              # Threads para baixar páginas (I/O-bound)
DB_BATCH_SIZE = 500           # Lote para inserção no banco de dados.
ENCODE_BATCH_SIZE = 128       # Lote para vetorização na GPU. Valor eficiente para LaBSE/MiniLM.

# --- Configurações do PGVector e dos Experimentos ---
BASE_TABLE_NAME = "normativos_cnj"

## MODELOS ATUALIZADOS: Foco em velocidade e eficiência ##
MODELS_TO_EVALUATE = [
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "short_name": "minilm"
    },
    {
        "name": "sentence-transformers/LaBSE",
        "dimension": 768,
        "short_name": "labse"
    },
    {
        "name": "BAAI/bge-m3",
        "dimension": 1024,
        "short_name": "bgem3"
    },
]

# --- Credenciais do Banco de Dados ---
load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME_POSTGRES = os.getenv("DB_NAME_POSTGRES")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

# --- Detecção de Hardware ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo de processamento: {DEVICE.upper()}")


# ==============================================================================
# PASSO 2: FUNÇÕES DE LIMPEZA E ESTRATÉGIAS DE CHUNKING
# (Nenhuma mudança necessária aqui)
# ==============================================================================
def slugify_column_name(name: str) -> str:
    """Limpa e padroniza um nome de coluna para ser seguro para SQL (slugify)."""
    if not name: return "coluna_desconhecida"
    text = unicodedata.normalize('NFD', str(name)).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text).strip('_')
    if not text: return "coluna_desconhecida"
    return text

def parse_and_clean_html_content(content_soup: BeautifulSoup) -> str:
    """Extrai texto puro de um objeto BeautifulSoup, removendo tags indesejadas."""
    if not content_soup: return ""
    for tag in content_soup(['script', 'style', 'a', 'img']):
        tag.decompose()
    for p in content_soup.find_all('p'):
        p.replace_with(p.get_text() + '\n')
    cleaned_text = content_soup.get_text(separator=' ', strip=True)
    return cleaned_text.replace('\n ', '\n').replace(' \n', '\n')

def clean_legal_text(text: str) -> str:
    """Aplica limpeza final em textos jurídicos para remover ruídos comuns."""
    if not text: return ""
    text = re.sub(r'^\s*[\.]{5,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\.{5,}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def chunk_recursive_langchain(raw_text: str, document_id: str, **kwargs) -> List[Dict]:
    """Divide o texto usando o RecursiveCharacterTextSplitter da LangChain, OTIMIZADO para texto jurídico."""
    splitter = RecursiveCharacterTextSplitter(
        # Separadores otimizados para texto jurídico
        separators=[
            # Tenta quebrar primeiro por estruturas maiores e mais significativas
            "\n\n",
            "\nArt.", # Quebra antes de um novo artigo
            "\nParágrafo",
            "\n§",
            # Depois por estruturas de frase
            ".\n",
            ". ",
            ";",
            ",",
            "\n",
            " "
        ], 
        chunk_size=600,       # Um tamanho de chunk um pouco menor e mais seguro
        chunk_overlap=60,     # Overlap proporcional
        length_function=len,
        is_separator_regex=False # Garante que os pontos não sejam tratados como regex
    )
    chunks_text = splitter.split_text(raw_text)
    final_chunks = []
    for i, chunk in enumerate(chunks_text):
        # Garante que chunks muito pequenos (restos de quebras) não sejam adicionados
        if len(chunk.strip()) > 50:
            final_chunks.append({
                "document_id": document_id,
                "autor": "Não identificado",
                "tipo": "Recursivo",
                "artigo_pai": f"parte_{i+1}",
                "chunk_text": chunk.strip()
            })
    return final_chunks

# Dicionário agora contém apenas a estratégia solicitada, simplificando o loop
CHUNKING_STRATEGIES = {"recursive": chunk_recursive_langchain}

# ==============================================================================
# PASSO 3: FUNÇÕES DE ACESSO À REDE E AO BANCO DE DADOS
# (Nenhuma mudança necessária aqui)
# ==============================================================================

def connect_to_postgres(host, port, db, user, pwd):
    """Cria e retorna uma conexão com o banco de dados PostgreSQL."""
    try:
        conn = psycopg2.connect(host=host, port=port, database=db, user=user, password=pwd)
        print(f"Conexão com PostgreSQL ({host}:{port}) bem-sucedida.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Erro ao conectar ao PostgreSQL: {e}")
        raise

def create_pg_table(conn, table_name: str, model_dimension: int):
    """Cria uma tabela no PostgreSQL com suporte a vetores, se ela não existir."""
    with conn.cursor() as cursor:
        print(f"Ativando extensão 'vector'...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        create_table_sql = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {} (
            id SERIAL PRIMARY KEY, document TEXT, metadata JSONB, embedding VECTOR({})
        );""").format(sql.Identifier(table_name), sql.Literal(model_dimension))
        cursor.execute(create_table_sql)
        print(f"Tabela '{table_name}' criada com sucesso.")

def batch_insert_data(conn, table_name: str, data_to_insert: List[tuple]):
    """Insere uma lista de dados (texto, metadados, vetor) no banco de dados de uma vez."""
    if not data_to_insert:
        return
    insert_sql = sql.SQL("INSERT INTO {} (document, metadata, embedding) VALUES %s").format(
        sql.Identifier(table_name)
    )
    with conn.cursor() as cursor:
        execute_values(cursor, insert_sql, data_to_insert)

def process_link(url: str, session: requests.Session, chunking_function: Callable) -> List[Dict]:
    """Baixa, parseia e chunkeia o conteúdo de uma única URL."""
    try:
        if DELAY_ENABLED:
            time.sleep(random.uniform(*DELAY_RANGE))
        response = session.get(url, timeout=30)
        response.raise_for_status()

        detailed_soup = BeautifulSoup(response.text, 'html.parser')
        main_div = detailed_soup.find('body').find('div')
        if not main_div: return []
            
        data_from_page = {}
        for id_div in main_div.find_all('div', class_='identificacao'):
            column_name = id_div.get_text(strip=True)
            content_div = id_div.find_next_sibling('div')
            if content_div:
                safe_col_name = slugify_column_name(column_name)
                cleaned_content = parse_and_clean_html_content(content_div)
                data_from_page[safe_col_name] = cleaned_content

        if not data_from_page: return []
            
        full_text = "\n".join(filter(None, data_from_page.values()))
        full_text = clean_legal_text(full_text)
        
        chunks = chunking_function(raw_text=full_text, document_id=url)

        for chunk in chunks:
            chunk['metadata'] = data_from_page.copy()
        
        return chunks

    except requests.exceptions.RequestException as e:
        print(f"Erro ao processar URL {url}: {e}")
        return []

# ==============================================================================
# PASSO 4: FUNÇÃO PRINCIPAL DE ORQUESTRAÇÃO DO SCRAPING
# ==============================================================================

def run_scraper(model: SentenceTransformer, conn, table_name: str, chunking_function: Callable):
    """Orquestra o processo de scraping e inserção de forma concorrente e em lote."""
    page_number = 406
    total_links_processed = 0
    session = requests.Session()

    while True:
        if PAGE_LIMIT > 0 and page_number > PAGE_LIMIT:
            print(f"\nLimite de {PAGE_LIMIT} página(s) atingido. Encerrando scraping.")
            break
        if LINK_LIMIT > 0 and total_links_processed >= LINK_LIMIT:
            print(f"\nLimite de {LINK_LIMIT} link(s) atingido. Encerrando.")
            break

        current_page_url = LISTING_URL.format(page_number)
        print(f"\n--- Buscando links na página: {current_page_url} ---")
        
        try:
            list_page_content = session.get(current_page_url, timeout=20).text
        except requests.exceptions.RequestException as e:
            print(f"Erro ao buscar a página de listagem: {e}. Pulando para a próxima.")
            page_number += 1
            continue

        soup = BeautifulSoup(list_page_content, 'html.parser')
        table_body = soup.find('table', class_='table').find('tbody') if soup.find('table') else None
        
        if not table_body or not table_body.find_all('tr'):
            print("Nenhuma linha encontrada. Fim da paginação.")
            break

        links_to_process = []
        for row in table_body.find_all('tr'):
            if LINK_LIMIT > 0 and (total_links_processed + len(links_to_process)) >= LINK_LIMIT:
                break
            link_tag = row.find('a', href=True)
            if link_tag:
                links_to_process.append(urljoin(ROOT_URL, link_tag['href']))

        if not links_to_process:
            print("Nenhum link novo encontrado nesta página.")
            page_number += 1
            continue

        all_chunks_from_page = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {executor.submit(process_link, url, session, chunking_function): url for url in links_to_process}
            
            progress = tqdm(as_completed(future_to_url), total=len(links_to_process), desc=f"Processando Links (Pág {page_number})")
            for future in progress:
                chunks = future.result()
                if chunks:
                    all_chunks_from_page.extend(chunks)
        
        total_links_processed += len(links_to_process)
        
        if not all_chunks_from_page:
            print("Nenhum chunk válido gerado a partir dos links desta página.")
            page_number += 1
            continue

        print(f"  |-> {len(all_chunks_from_page)} chunks gerados de {len(links_to_process)} links.")
        
        print(f"  |-> Gerando embeddings para {len(all_chunks_from_page)} chunks (Batch Size: {ENCODE_BATCH_SIZE})...")
        texts_to_embed = [chunk['chunk_text'] for chunk in all_chunks_from_page]
        
        ## Codificação direta, sem a complexidade do FP16 que não é necessária para estes modelos.
        embeddings = model.encode(
            texts_to_embed, 
            show_progress_bar=True, 
            batch_size=ENCODE_BATCH_SIZE
        )
        
        data_for_db = []
        for i, chunk in enumerate(all_chunks_from_page):
            metadata_to_insert = chunk.get('metadata', {})
            metadata_to_insert.pop("texto", None) # Remove campo de texto duplicado se existir
            metadata_to_insert.update({
                "document_id": chunk.get("document_id"),
                "autor": chunk.get("autor"),
                "tipo_chunk": chunk.get("tipo"),
                "artigo_pai": chunk.get("artigo_pai")
            })
            metadata_json = json.dumps(metadata_to_insert, ensure_ascii=False)
            
            data_for_db.append(
                (chunk['chunk_text'], metadata_json, embeddings[i].tolist())
            )

        print(f"  |-> Inserindo {len(data_for_db)} registros no banco de dados (Lotes de {DB_BATCH_SIZE})...")
        total_inserted = 0
        try:
            for i in range(0, len(data_for_db), DB_BATCH_SIZE):
                batch = data_for_db[i:i + DB_BATCH_SIZE]
                batch_insert_data(conn, table_name, batch)
                total_inserted += len(batch)
            
            conn.commit()
            print(f"  |-> Commit realizado. {total_inserted} registros salvos na tabela '{table_name}'.")
        except Exception as e:
            print(f"  Erro na inserção em lote: {e}")
            conn.rollback()

        page_number += 1

# ==============================================================================
# PASSO 5: ORQUESTRADOR DE EXPERIMENTOS
# ==============================================================================

def main():
    """Função principal que gerencia a conexão e orquestra os experimentos."""
    conn = None
    try:
        conn = connect_to_postgres(DB_HOST, DB_PORT, DB_NAME_POSTGRES, DB_USER, DB_PASS)
        if not conn:
            raise ConnectionError("Não foi possível conectar ao banco de dados.")

        for model_config in MODELS_TO_EVALUATE:
            model_name = model_config['name']
            model_short_name = model_config['short_name']
            model_dimension = model_config['dimension']
            
            print(f"\n{'='*25} INICIANDO EXPERIMENTO COM O MODELO: {model_name.upper()} {'='*25}")
            
            print(f"Carregando modelo '{model_name}' para o dispositivo '{DEVICE}'...")
            model = SentenceTransformer(model_name, device=DEVICE)
            print("Modelo carregado com sucesso.")

            # O loop agora itera sobre um único item, mas a estrutura é mantida para flexibilidade futura.
            for strategy_name, chunking_function in CHUNKING_STRATEGIES.items():
                
                table_name = f"{BASE_TABLE_NAME}_{model_short_name}_{strategy_name}"
                
                print(f"\n--- Executando: Modelo='{model_short_name}', Estratégia='{strategy_name}', Tabela='{table_name}' ---")
                
                create_pg_table(conn, table_name, model_dimension)
                conn.commit()
                
                run_scraper(model, conn, table_name, chunking_function)
                print(f"--- Experimento CONCLUÍDO: Modelo='{model_short_name}', Estratégia='{strategy_name}' ---")

        print(f"\n{'='*25} TODOS OS EXPERIMENTOS FORAM CONCLUÍDOS {'='*25}")

    except Exception as e:
        print(f"\nOcorreu um erro crítico no processo: {e}")
        if conn:
            print("Revertendo transação (rollback)...")
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("\nConexão com PostgreSQL fechada.")

if __name__ == "__main__":
    main()