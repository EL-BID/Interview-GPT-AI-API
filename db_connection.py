import os
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def get_db_connection():
    """
    Función auxiliar que obtiene el checkpointer asíncrono para PostgreSQL.
    """
    try:
        # Construir la cadena de conexión
        conn_string = (
            f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
            f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
            f"?sslmode={os.getenv('POSTGRES_SSLMODE', 'require')}"
        )
        
        print(f"Intentando conectar a: {conn_string}")
        
        # Crear un pool de conexiones asíncrono
        pool = AsyncConnectionPool(
            conninfo=conn_string,
            max_size=20,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        ) 
            # Crear el checkpointer asíncrono
        checkpointer = AsyncPostgresSaver(pool)
            
       
            
        return checkpointer, pool
        
    except Exception as e:
        print(f"Error en get_db_connection: {str(e)}")
        raise 