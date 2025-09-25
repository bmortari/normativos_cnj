# main_optimized.py

#!/usr/bin/env python3
"""
Script OTIMIZADO para Web Scraping, Processamento e Vetorização de Atos Normativos do CNJ.
... (resto da docstring original) ...
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
from concurrent.futures import ThreadPoolExecutor, as_completed ## OTIMIZAÇÃO ##

# --- Bibliotecas de Terceiros ---
import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values ## OTIMIZAÇÃO ##
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm

# ==============================================================================
# PASSO 1: CONFIGURAÇÕES GERAIS E DE EXPERIMENTOS
# ==============================================================================

# --- Configurações do Scraping ---
ROOT_URL = 'https://atos.cnj.jus.br'
LISTING_URL = ROOT_URL + '/atos?atos=sim&page={}'
PAGE_LIMIT = 2      # Limite de páginas para um teste rápido (0 para todas)
LINK_LIMIT = 50     # Limite total de links por experimento para um teste rápido (0 para todos)
DELAY_ENABLED = False # Desabilitar delay para máxima performance em testes. Reativar para scraping massivo.
DELAY_RANGE = (0.1, 0.3) 

## OTIMIZAÇÃO: Configurações para paralelismo e lotes ##
MAX_WORKERS = 10              # Número de threads para baixar páginas simultaneamente
DB_BATCH_SIZE = 100           # Quantidade de registros para inserir no banco de dados de uma vez

# --- Configurações do PGVector e dos Experimentos ---
BASE_TABLE_NAME = "normativos_cnj" 

MODELS_TO_EVALUATE = [
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2", # Colocado primeiro por ser mais leve para testar
        "dimension": 384,
        "short_name": "minilm"
    },
    {
        "name": "sentence-transformers/LaBSE",
        "dimension": 768,
        "short_name": "labse"
    },
    {
        "name": "alfaneo/bertimbau-base-portuguese-sts",
        "dimension": 768,
        "short_name": "bertimbau"
    }
]

# --- Credenciais do Banco de Dados ---
load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME_POSTGRES = os.getenv("DB_NAME_POSTGRES")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")


# ==============================================================================
# PASSO 2: FUNÇÕES DE LIMPEZA E ESTRATÉGIAS DE CHUNKING
# (Nenhuma mudança necessária aqui, as funções já são eficientes)
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

def chunk_structured_legal_text(raw_text: str, document_id: str, **kwargs) -> List[Dict]:
    """Divide o texto legal usando uma abordagem de 'scanner' baseada em marcadores jurídicos."""
    final_chunks = []
    autor = "Não identificado"
    signature_pattern = r'[\n\s]+([A-Z\s]{5,})\s*?\n\s*?(Ministro|Presidente|Corregedor|Relator|Conselheiro)[\s\S]*'
    signature_match = re.search(signature_pattern, raw_text, re.MULTILINE | re.IGNORECASE)
    if signature_match:
        autor_block = signature_match.group(0).strip()
        autor = autor_block.split('\n')[0].strip()
        raw_text = raw_text[:signature_match.start()].strip()
    master_pattern = re.compile(r'(CONSIDERANDO|RESOLVE:|DETERMINA:|Art\.\s*\d+º(?:-A|-B)?\.?|Parágrafo\s+único\.?|§\s*\d+º\.?)', re.IGNORECASE)
    matches = list(master_pattern.finditer(raw_text))
    if not matches:
        if raw_text.strip():
            final_chunks.append({"document_id": document_id, "autor": autor, "tipo": "Corpo_Unico", "artigo_pai": None, "chunk_text": raw_text.strip()})
        return final_chunks
    first_match_start = matches[0].start()
    if first_match_start > 0:
        preambulo_text = raw_text[:first_match_start].strip()
        if preambulo_text:
            final_chunks.append({"document_id": document_id, "autor": autor, "tipo": "Preambulo", "artigo_pai": None, "chunk_text": preambulo_text})
    artigo_pai_atual = None
    for i, match in enumerate(matches):
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        chunk_text = raw_text[start_pos:end_pos].strip()
        marker_text = match.group(1).strip()
        tipo_chunk = "Desconhecido"
        if 'art' in marker_text.lower():
            tipo_chunk = "Artigo"
            artigo_pai_atual = re.match(r'^(Art\.\s*\d+º(?:-A|-B)?)', marker_text, re.IGNORECASE).group(1)
        elif 'considerando' in marker_text.lower(): tipo_chunk = "Considerando"
        elif 'parágrafo' in marker_text.lower() or '§' in marker_text: tipo_chunk = "Paragrafo"
        elif 'resolve' in marker_text.lower() or 'determina' in marker_text.lower():
            tipo_chunk = "Resolucao"
            if i + 1 < len(matches): continue
        if chunk_text:
            final_chunks.append({"document_id": document_id, "autor": autor, "tipo": tipo_chunk, "artigo_pai": artigo_pai_atual, "chunk_text": chunk_text})
    MIN_CHUNK_SIZE = 30
    return [chunk for chunk in final_chunks if len(chunk['chunk_text']) > MIN_CHUNK_SIZE]

def chunk_fixed_size(raw_text: str, document_id: str, chunk_size=1024, chunk_overlap=100, **kwargs) -> List[Dict]:
    """Divide o texto em pedaços de tamanho fixo com sobreposição."""
    final_chunks = []
    start = 0
    text_len = len(raw_text)
    chunk_id = 0
    while start < text_len:
        end = start + chunk_size
        chunk_text = raw_text[start:end]
        if chunk_text.strip():
            final_chunks.append({"document_id": document_id, "autor": "Não identificado", "tipo": "Fixo", "artigo_pai": f"chunk_{chunk_id}", "chunk_text": chunk_text})
        start += chunk_size - chunk_overlap
        chunk_id += 1
    return final_chunks

CHUNKING_STRATEGIES = {"structured": chunk_structured_legal_text, "fixed": chunk_fixed_size}

# ==============================================================================
# PASSO 3: FUNÇÕES DE ACESSO À REDE E AO BANCO DE DADOS
# (Modificadas para suportar lotes e sessões)
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
        print(f"Preparando para (re)criar a tabela '{table_name}'...")
        cursor.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name)))
        create_table_sql = sql.SQL("""
        CREATE TABLE {} (
            id SERIAL PRIMARY KEY, document TEXT, metadata JSONB, embedding VECTOR({})
        );""").format(sql.Identifier(table_name), sql.Literal(model_dimension))
        cursor.execute(create_table_sql)
        print(f"Tabela '{table_name}' criada com sucesso.")

## OTIMIZAÇÃO: Função de inserção em lote ##
def batch_insert_data(conn, table_name: str, data_to_insert: List[tuple]):
    """Insere uma lista de dados (texto, metadados, vetor) no banco de dados de uma vez."""
    if not data_to_insert:
        return
    
    insert_sql = sql.SQL("INSERT INTO {} (document, metadata, embedding) VALUES %s").format(
        sql.Identifier(table_name)
    )
    with conn.cursor() as cursor:
        execute_values(cursor, insert_sql, data_to_insert)

## OTIMIZAÇÃO: Função para processar um único link (para ser usada em threads) ##
def process_link(url: str, session: requests.Session, chunking_function: Callable) -> List[Dict]:
    """
    Baixa, parseia e chunkeia o conteúdo de uma única URL.
    Retorna uma lista de chunks ou uma lista vazia em caso de erro.
    """
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

        # Adiciona metadados completos a cada chunk
        for chunk in chunks:
            chunk['metadata'] = data_from_page.copy()
        
        return chunks

    except requests.exceptions.RequestException as e:
        print(f"Erro ao processar URL {url}: {e}")
        return []

# ==============================================================================
# PASSO 4: FUNÇÃO PRINCIPAL DE ORQUESTRAÇÃO DO SCRAPING
# (Completamente reescrita para ser concorrente e em lote)
# ==============================================================================

def run_scraper(model: SentenceTransformer, conn, table_name: str, chunking_function: Callable):
    """Orquestra o processo de scraping e inserção de forma concorrente e em lote."""
    page_number = 1
    total_links_processed = 0
    session = requests.Session() ## OTIMIZAÇÃO: Usa uma sessão para reutilizar conexões

    while True:
        if PAGE_LIMIT > 0 and page_number > PAGE_LIMIT:
            print(f"\nLimite de {PAGE_LIMIT} página(s) atingido. Encerrando scraping.")
            break
        if LINK_LIMIT > 0 and total_links_processed >= LINK_LIMIT:
            print(f"\nLimite de {LINK_LIMIT} link(s) atingido. Encerrando.")
            break

        current_page_url = LISTING_URL.format(page_number)
        print(f"\n--- Buscando links na página: {current_page_url} ---")
        
        list_page_content = session.get(current_page_url).text
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
        # ## OTIMIZAÇÃO: Processamento concorrente dos links ##
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
        
        # ## OTIMIZAÇÃO: Geração de embeddings em lote ##
        print(f"  |-> Gerando embeddings para {len(all_chunks_from_page)} chunks de uma vez...")
        texts_to_embed = [chunk['chunk_text'] for chunk in all_chunks_from_page]
        embeddings = model.encode(texts_to_embed, show_progress_bar=True)
        
        # ## OTIMIZAÇÃO: Preparação dos dados para inserção em lote ##
        data_for_db = []
        for i, chunk in enumerate(all_chunks_from_page):
            metadata_to_insert = chunk['metadata']
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

        # ## OTIMIZAÇÃO: Inserção no banco de dados em lotes ##
        print(f"  |-> Inserindo {len(data_for_db)} registros no banco de dados em lotes...")
        try:
            batch_insert_data(conn, table_name, data_for_db)
            conn.commit()
            print(f"  |-> Commit realizado. {len(data_for_db)} registros salvos na tabela '{table_name}'.")
        except Exception as e:
            print(f"  Erro na inserção em lote: {e}")
            conn.rollback()

        page_number += 1

# ==============================================================================
# PASSO 5: ORQUESTRADOR DE EXPERIMENTOS
# (Pequenas modificações para se adequar ao novo fluxo)
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
            
            print(f"\n{'='*25} INICIANDO EXPERIMENTOS COM O MODELO: {model_name.upper()} {'='*25}")
            
            print(f"Carregando modelo '{model_name}'...")
            model = SentenceTransformer(model_name)
            print("Modelo carregado com sucesso.")

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