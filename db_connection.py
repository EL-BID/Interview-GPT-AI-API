import os
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def get_db_connection():
    """
    Helper function that gets the asynchronous checkpointer for PostgreSQL.
    """
    try:
        # Build connection string
        conn_string = (
            f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
            f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
            f"?sslmode={os.getenv('POSTGRES_SSLMODE', 'prefer')}"
        )
        
        print(f"Attempting to connect to: {conn_string}")
        
        # Create an asynchronous connection pool
        pool = AsyncConnectionPool(
            conninfo=conn_string,
            max_size=20,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
                "connect_timeout": 10,  # Add connection timeout
            },
        )
        
        # Create the asynchronous checkpointer
        checkpointer = AsyncPostgresSaver(pool)
        
        return checkpointer, pool
        
    except Exception as e:
        print(f"Error in get_db_connection: {str(e)}")
        raise 