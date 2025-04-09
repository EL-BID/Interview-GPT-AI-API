import os
import asyncio
import logging
from dotenv import load_dotenv
from db_connection import get_db_connection

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

async def test_connection():
    """
    Función para probar la conexión a la base de datos y verificar si se crea la tabla.
    """
    try:
        logger.info("Iniciando prueba de conexión a la base de datos...")
        
        # Obtener conexión a la base de datos
        logger.info("Obteniendo conexión a la base de datos...")
        conn = await get_db_connection()
        logger.info("Conexión a la base de datos establecida")
        
        try:
            # Verificar si la tabla existe
            logger.info("Verificando si las tablas existen...")
            async with conn.cursor() as cursor:
                # Consultar las tablas en la base de datos
                await cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                tables = await cursor.fetchall()
                
                logger.info("Tablas encontradas:")
                for table in tables:
                    logger.info(f"- {table[0]}")
                
                # Insertar un registro de prueba
                logger.info("Insertando registro de prueba en langgraph_checkpoints...")
                await cursor.execute("""
                    INSERT INTO langgraph_checkpoints (key, value, config)
                    VALUES ('test-key', '{"test": "data"}'::jsonb, '{"thread_id": "test-thread"}'::jsonb)
                    RETURNING id;
                """)
                result = await cursor.fetchone()
                logger.info(f"Registro insertado con ID: {result[0]}")
                
                # Verificar el registro insertado
                logger.info("Verificando el registro insertado...")
                await cursor.execute("""
                    SELECT * FROM langgraph_checkpoints WHERE key = 'test-key';
                """)
                checkpoint = await cursor.fetchone()
                logger.info(f"Registro encontrado: {checkpoint}")
                
                await conn.commit()
                logger.info("Prueba completada con éxito")
            
        finally:
            # Cerrar la conexión
            logger.info("Cerrando conexión a la base de datos...")
            await conn.close()
            logger.info("Conexión a la base de datos cerrada")
        
    except Exception as e:
        logger.error(f"Error en la prueba de conexión: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_connection()) 