"""
Módulo para operações de banco de dados PostgreSQL com suporte a vetores.
"""

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from typing import List, Tuple


def connect_to_postgres(host: str, port: str, db: str, user: str, pwd: str):
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


def batch_insert_data(conn, table_name: str, data_to_insert: List[Tuple]):
    """Insere uma lista de dados (texto, metadados, vetor) no banco de dados de uma vez."""
    if not data_to_insert:
        return
    
    insert_sql = sql.SQL("INSERT INTO {} (document, metadata, embedding) VALUES %s").format(
        sql.Identifier(table_name)
    )
    with conn.cursor() as cursor:
        execute_values(cursor, insert_sql, data_to_insert)
