#!/usr/bin/env python3
"""
Script OTIMIZADO para Web Scraping, Processamento e Vetorização de Atos Normativos do CNJ.

Este script principal orquestra todo o processo de:
1. Web scraping de atos normativos do site do CNJ
2. Processamento e limpeza dos textos jurídicos
3. Divisão dos textos em chunks usando diferentes estratégias
4. Geração de embeddings usando modelos de sentence transformers
5. Armazenamento no PostgreSQL com suporte a vetores (PGVector)

Uso:
    python main.py

Requisitos:
    - PostgreSQL com extensão vector instalada
    - Arquivo .env com credenciais do banco de dados
    - Dependências listadas no requirements.txt
"""

from sentence_transformers import SentenceTransformer

from src.normativos_cnj.config import (
    DB_HOST, DB_PORT, DB_NAME_POSTGRES, DB_USER, DB_PASS,
    MODELS_TO_EVALUATE, BASE_TABLE_NAME
)
from src.normativos_cnj.database import connect_to_postgres, create_pg_table
from src.normativos_cnj.scraper import run_scraper
from src.normativos_cnj.text_processing import CHUNKING_STRATEGIES


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