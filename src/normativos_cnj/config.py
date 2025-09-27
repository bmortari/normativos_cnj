"""
Configurações do projeto para Web Scraping, Processamento e Vetorização de Atos Normativos do CNJ.
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# --- Configurações do Scraping ---
ROOT_URL = 'https://atos.cnj.jus.br'
LISTING_URL = ROOT_URL + '/atos?atos=sim&page={}'
PAGE_LIMIT = 2      # Limite de páginas para um teste rápido (0 para todas)
LINK_LIMIT = 50     # Limite total de links por experimento para um teste rápido (0 para todos)
DELAY_ENABLED = False # Desabilitar delay para máxima performance em testes. Reativar para scraping massivo.
DELAY_RANGE = (0.1, 0.3) 

# --- Configurações de Performance ---
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
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME_POSTGRES = os.getenv("DB_NAME_POSTGRES")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
