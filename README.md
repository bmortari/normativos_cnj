# Normativos CNJ

Web Scraping, Processamento e Vetorização de Atos Normativos do CNJ.

## Estrutura do Projeto

```
normativos_cnj/
├── src/
│   └── normativos_cnj/
│       ├── __init__.py
│       ├── config.py          # Configurações do projeto
│       ├── database.py         # Operações de banco de dados
│       ├── scraper.py          # Web scraping
│       ├── text_processing.py  # Processamento de texto
│       └── embeddings.py       # Geração de embeddings
├── main.py                     # Script principal
├── pyproject.toml             # Configuração do projeto
├── requirements.txt           # Dependências
└── README.md                  # Este arquivo
```

## Instalação

### Usando pip (recomendado)

```bash
pip install -e .
```

### Usando requirements.txt

```bash
pip install -r requirements.txt
```

## Configuração

1. Crie um arquivo `.env` na raiz do projeto com as credenciais do banco de dados:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME_POSTGRES=normativos_cnj
DB_USER=seu_usuario
DB_PASS=sua_senha
```

2. Certifique-se de que o PostgreSQL está rodando e tem a extensão `vector` instalada:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Uso

Execute o script principal:

```bash
python main.py
```

## Módulos

### config.py
Contém todas as configurações do projeto, incluindo:
- URLs do site do CNJ
- Configurações de scraping
- Modelos de embeddings
- Credenciais do banco de dados

### database.py
Módulo responsável pelas operações de banco de dados:
- Conexão com PostgreSQL
- Criação de tabelas com suporte a vetores
- Inserção em lote de dados

### scraper.py
Módulo de web scraping:
- Download de páginas de atos normativos
- Processamento concorrente de links
- Extração de metadados

### text_processing.py
Módulo de processamento de texto:
- Limpeza de HTML
- Normalização de texto jurídico
- Estratégias de chunking (estruturado e tamanho fixo)

### embeddings.py
Módulo de geração de embeddings:
- Geração de embeddings em lote
- Preparação de dados para inserção no banco

## Desenvolvimento

### Instalação em modo desenvolvimento

```bash
pip install -e ".[dev]"
```

### Executando testes

```bash
pytest
```

### Formatação de código

```bash
black src/
isort src/
```

### Verificação de tipos

```bash
mypy src/
```

## Licença

MIT License
