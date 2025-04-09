import azure.functions as func
import logging

app = func.FunctionApp()

@app.function_name(name="test")
@app.route(route="test", methods=["GET"])
def test_function(req: func.HttpRequest) -> func.HttpResponse:
    """
    Función de prueba simple.
    """
    logging.info('Función de prueba ejecutada')
    return func.HttpResponse(
        "¡La función de prueba funciona correctamente!",
        mimetype="text/plain",
        status_code=200
    ) 