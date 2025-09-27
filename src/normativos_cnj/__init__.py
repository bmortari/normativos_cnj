"""
Pacote para Web Scraping, Processamento e Vetorização de Atos Normativos do CNJ.

Este pacote contém módulos para:
- Configurações do projeto
- Web scraping de atos normativos
- Processamento e limpeza de textos jurídicos
- Geração de embeddings
- Operações de banco de dados PostgreSQL com suporte a vetores
"""

from . import config
from . import database
from . import scraper
from . import text_processing
from . import embeddings

__version__ = "1.0.0"
__author__ = "CNJ Normativos Team"
