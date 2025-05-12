import os
import graphviz
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from interview_flow import build_graph

def visualize_interview_graph():
    """
    Visualiza el grafo de la entrevista usando Graphviz.
    """
    try:
        # Obtener el grafo
        graph = build_graph()
        
        # Generar la visualización Mermaid
        mermaid_syntax = graph.get_graph().draw_mermaid()
        print("Sintaxis Mermaid del grafo:")
        print(mermaid_syntax)
        
        # Crear un grafo Graphviz
        dot = graphviz.Digraph('Interview Graph', comment='Grafo de la Entrevista')
        dot.attr(rankdir='TB')  # Top to Bottom
        dot.attr('node', shape='box', style='filled', fillcolor='lightblue')
        
        # Definir los nodos
        dot.node('start', 'START', shape='circle', fillcolor='#ffdfba')
        dot.node('validate', 'validate_response', fillcolor='#fad7de')
        dot.node('interviewer', 'interviewer', fillcolor='#fad7de')
        dot.node('despedida', 'despedida', fillcolor='#fad7de')
        dot.node('end', 'END', shape='circle', fillcolor='#baffc9')
        
        # Definir las conexiones
        dot.edge('start', 'validate')
        dot.edge('validate', 'interviewer', label='is_complete=False')
        dot.edge('validate', 'despedida', label='is_complete=True')
        dot.edge('interviewer', 'end')
        dot.edge('despedida', 'end')
        
        # Guardar y renderizar el grafo
        dot.render('interview_graph', format='png', cleanup=True)
        print(f"\nGrafo guardado en: {os.path.abspath('interview_graph.png')}")
        
        # Mostrar la imagen
        display(Image(filename='interview_graph.png'))
        
    except Exception as e:
        print(f"Error al visualizar el grafo: {str(e)}")
        print("\nSugerencias para resolver el error:")
        print("1. Asegúrate de tener Graphviz instalado en tu sistema:")
        print("   - En macOS: brew install graphviz")
        print("   - En Ubuntu/Debian: sudo apt-get install graphviz")
        print("   - En Windows: Descarga e instala desde https://graphviz.org/download/")
        print("2. Instala la biblioteca Python de Graphviz:")
        print("   pip install graphviz")

if __name__ == "__main__":
    visualize_interview_graph() 