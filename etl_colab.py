# re_embed.py

#!/usr/bin/env python3
"""
Script OTIMIZADO para ler chunks de texto existentes de tabelas no PostgreSQL,
gerar novos embeddings com o modelo BAAI/bge-m3, e salvar em novas tabelas.
Ideal para rodar em ambientes com GPU como o Google Colab.
"""

# --- Bibliotecas Padrão do Python ---
import os
import json
from typing import List

# --- Bibliotecas de Terceiros ---
# No Colab, instale com:
# !pip install psycopg2-binary sentence-transformers python-dotenv tqdm torch
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm
import torch

# ==============================================================================
# PASSO 1: CONFIGURAÇÕES GERAIS
# ==============================================================================

# --- Configurações das Tabelas de Origem e Destino ---
# Defina aqui as tabelas que você quer processar.
# O script irá ler de 'source_table' e criar/escrever em uma nova tabela.
SOURCES_TO_PROCESS = [
    {
        "source_table": "normativos_cnj_minilm_recursive",
        "strategy": "recursive"
    },
]

# --- Configurações do Modelo ---
MODEL_CONFIG = {
        "name": "sentence-transformers/LaBSE",
        "dimension": 768,
        "short_name": "labse"
    }

# --- Configurações de Otimização ---
DB_FETCH_BATCH_SIZE = 1000    # Quantos registros ler do banco de dados por vez.
ENCODE_BATCH_SIZE = 64       # Lote para vetorização na GPU. Se der erro de memória, reduza para 64.
DB_INSERT_BATCH_SIZE = 1000   # Quantos registros inserir no banco de dados por vez.

# --- Credenciais do Banco de Dados ---
load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME_POSTGRES = os.getenv("DB_NAME_POSTGRES")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

# --- Detecção de Hardware e Otimizações de GPU ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FP16_ENABLED = DEVICE == 'cuda' and torch.cuda.get_device_capability(0)[0] >= 7

print(f"Dispositivo de processamento: {DEVICE.upper()}")
print(f"Mixed Precision (FP16) ativado: {FP16_ENABLED}")

# ==============================================================================
# PASSO 2: FUNÇÕES DE BANCO DE DADOS
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
    """Cria a tabela de destino para os novos embeddings."""
    with conn.cursor() as cursor:
        print(f"Ativando extensão 'vector'...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Apaga a tabela se ela já existir para garantir uma execução limpa
        cursor.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(sql.Identifier(table_name)))
        print(f"Tabela antiga '{table_name}' removida (se existia).")
        
        create_table_sql = sql.SQL("""
        CREATE TABLE {} (
            id SERIAL PRIMARY KEY,
            document TEXT,
            metadata JSONB,
            embedding VECTOR({})
        );""").format(sql.Identifier(table_name), sql.Literal(model_dimension))
        cursor.execute(create_table_sql)
        conn.commit()
        print(f"Tabela de destino '{table_name}' criada com sucesso.")

def batch_insert_data(conn, table_name: str, data_to_insert: List[tuple]):
    """Insere uma lista de dados (texto, metadados, vetor) no banco de dados de uma vez."""
    if not data_to_insert:
        return
    # Note que não inserimos o 'id', ele será gerado automaticamente (SERIAL)
    insert_sql = sql.SQL("INSERT INTO {} (document, metadata, embedding) VALUES %s").format(
        sql.Identifier(table_name)
    )
    with conn.cursor() as cursor:
        execute_values(cursor, insert_sql, data_to_insert)

# ==============================================================================
# PASSO 3: LÓGICA PRINCIPAL DE RE-VETORIZAÇÃO
# ==============================================================================

def re_embed_table(conn, model: SentenceTransformer, source_table: str, new_table_name: str):
    """
    Lê dados de uma tabela de origem, gera novos embeddings e insere em uma nova tabela.
    Processa os dados em lotes para otimizar o uso de memória.
    """
    with conn.cursor() as cursor:
        cursor.execute(sql.SQL("SELECT COUNT(*) FROM {};").format(sql.Identifier(source_table)))
        total_rows = cursor.fetchone()[0]
        if total_rows == 0:
            print(f"Tabela de origem '{source_table}' está vazia. Pulando.")
            return
    print(f"Encontrados {total_rows} registros em '{source_table}'. Iniciando processo...")

    offset = 0
    with tqdm(total=total_rows, desc=f"Processando {source_table}") as pbar:
        while offset < total_rows:
            # 1. LER um lote do banco de dados
            with conn.cursor() as cursor:
                query = sql.SQL("SELECT document, metadata FROM {} ORDER BY id LIMIT %s OFFSET %s;").format(
                    sql.Identifier(source_table)
                )
                cursor.execute(query, (DB_FETCH_BATCH_SIZE, offset))
                records = cursor.fetchall()

            if not records:
                break

            texts_to_embed = [rec[0] for rec in records]
            
            # 2. VETORIZAR o lote com otimização de GPU
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=FP16_ENABLED):
                embeddings = model.encode(
                    texts_to_embed,
                    show_progress_bar=False, # A barra de progresso principal já controla
                    batch_size=ENCODE_BATCH_SIZE
                )
            
            # 3. PREPARAR dados para inserção
            data_for_db = []
            for i, record in enumerate(records):
                document_text = record[0]
                metadata = record[1]
                embedding_vector = embeddings[i].tolist()
                data_for_db.append((document_text, json.dumps(metadata), embedding_vector))

            # 4. INSERIR o lote no novo banco de dados
            try:
                batch_insert_data(conn, new_table_name, data_for_db)
                conn.commit()
            except Exception as e:
                print(f"Erro na inserção em lote: {e}")
                conn.rollback()
                # Opcional: decidir se quer parar ou continuar em caso de erro
                # raise e 

            offset += len(records)
            pbar.update(len(records))

# ==============================================================================
# PASSO 4: ORQUESTRADOR
# ==============================================================================

def main():
    """Função principal que gerencia a conexão e orquestra o processo."""
    conn = None
    try:
        conn = connect_to_postgres(DB_HOST, DB_PORT, DB_NAME_POSTGRES, DB_USER, DB_PASS)
        if not conn:
            raise ConnectionError("Não foi possível conectar ao banco de dados.")

        print(f"\nCarregando modelo '{MODEL_CONFIG['name']}' para o dispositivo '{DEVICE}'...")
        model = SentenceTransformer(MODEL_CONFIG['name'], device=DEVICE)
        print("Modelo carregado com sucesso.")

        for task in SOURCES_TO_PROCESS:
            source_table = task['source_table']
            strategy = task['strategy']
            new_table_name = f"normativos_cnj_{MODEL_CONFIG['short_name']}_{strategy}"
            
            print(f"\n{'='*25} INICIANDO TAREFA {'='*25}")
            print(f"Origem: '{source_table}'")
            print(f"Destino: '{new_table_name}'")
            print(f"{'='*53}")

            # Cria a tabela de destino (e limpa se já existir)
            create_pg_table(conn, new_table_name, MODEL_CONFIG['dimension'])
            
            # Executa o processo de re-vetorização
            re_embed_table(conn, model, source_table, new_table_name)
            
            print(f"--- Tarefa CONCLUÍDA para a tabela '{source_table}' ---")

        print(f"\n{'='*25} TODO O PROCESSO FOI CONCLUÍDO {'='*25}")

    except Exception as e:
        print(f"\nOcorreu um erro crítico no processo: {e}")
        if conn:
            print("Revertendo transação pendente (rollback)...")
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("\nConexão com PostgreSQL fechada.")