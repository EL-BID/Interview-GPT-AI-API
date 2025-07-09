# Interview GPT AI API

**Un servicio API con capacidades de IA construido con Azure Functions, LangGraph y PostgreSQL que proporciona gestión inteligente de flujo de conversación y validación de respuestas para la aplicación Interview GPT.**

---

## Tabla de Contenidos

1. [¿Qué es Interview GPT AI API?](#qué-es-interview-gpt-ai-api)
2. [Características Principales](#características-principales)
3. [Arquitectura del Sistema](#arquitectura-del-sistema)
4. [Guía de Instalación](#guía-de-instalación)
   - [4.1 Requisitos del Sistema](#41-requisitos-del-sistema)
   - [4.2 Configuración del Entorno](#42-configuración-del-entorno)
   - [4.3 Instalación de Dependencias](#43-instalación-de-dependencias)
   - [4.4 Ejecución en Desarrollo](#44-ejecución-en-desarrollo)
5. [Documentación de la API](#documentación-de-la-api)
   - [5.1 Endpoints de Gestión de Entrevistas LangGraph](#51-endpoints-de-gestión-de-entrevistas-langgraph)
   - [5.2 Endpoints de Chat AI General](#52-endpoints-de-chat-ai-general)
6. [Arquitectura Técnica](#arquitectura-técnica)
   - [6.1 Flujo de Entrevista](#61-flujo-de-entrevista)
   - [6.2 Configuración de Base de Datos](#62-configuración-de-base-de-datos)
   - [6.3 Filtrado de Contenido](#63-filtrado-de-contenido)
7. [Despliegue en Producción](#despliegue-en-producción)
8. [Contribuciones](#contribuciones)
9. [Autores](#autores)

---

## ¿Qué es Interview GPT AI API?

Interview GPT AI API es un servicio con capacidades de IA diseñado para proporcionar capacidades de entrevistas conversacionales inteligentes para la aplicación Interview GPT. El sistema aprovecha las capacidades de IA para proporcionar:

- **Flujo de Entrevista Inteligente**: Proceso de entrevista automatizado con preguntas conscientes del contexto usando LangGraph
- **Validación de Respuestas**: Validación impulsada por IA para asegurar respuestas completas y relevantes
- **Soporte Multiidioma**: Las entrevistas se pueden realizar en múltiples idiomas con detección dinámica de idioma
- **Persistencia de Conversación**: Sistema de checkpoints basado en PostgreSQL para gestión del estado de conversación
- **Manejo de Filtros de Contenido**: Filtrado automático de contenido y reformulación de mensajes para interacciones apropiadas
- **Gestión de Entrevistas**: Gestión completa del ciclo de vida de entrevistas con checkpoints

La plataforma está construida con una arquitectura de microservicios usando Azure Functions.

---

## Características Principales

- **Flujo de Conversación Inteligente**: Flujo de trabajo basado en LangGraph con preguntas de seguimiento automatizadas
- **Validación de Respuestas**: Evaluación de completitud y validación de contexto impulsada por IA
- **Soporte Multiidioma**: Detección dinámica de idioma y mantenimiento consistente del idioma
- **Persistencia de Conversación**: Checkpoints de PostgreSQL para recuperación de conversación y gestión de estado
- **Manejo de Filtros de Contenido**: Detección automática y reformulación de contenido inapropiado
- **Gestión de Entrevistas**: Ciclo de vida completo de entrevistas con gestión de checkpoints

---

## Arquitectura del Sistema

- **Backend**: Azure Functions con Python 
- **IA/ML**: Azure OpenAI con LangGraph para flujo de conversación inteligente
- **Base de Datos**: PostgreSQL con agrupación de conexiones asíncronas para persistencia de datos
- **Gestión de Estado**: Checkpoints de LangGraph para persistencia del estado de conversación
- **Filtrado de Contenido**: Filtrado de contenido con capacidades de reformulación automática

---

## Guía de Instalación

### 4.1 Requisitos del Sistema

- **Python** 3.8 o superior
- **Azure Functions Core Tools** para desarrollo local
- **PostgreSQL** base de datos (versión 12 o superior)
- **Azure OpenAI** servicio con acceso API apropiado

### 4.2 Configuración del Entorno

Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:

```env
# Configuración de Azure OpenAI
AZURE_OPEN_AI_ENDPOINT=tu_endpoint_azure_openai
AZURE_OPENAI_API_KEY=tu_clave_api_azure_openai
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_API_INSTANCE_NAME=tu_nombre_instancia
AZURE_OPENAI_API_BASE_PATH=tu_ruta_base
AZURE_DEPLOYMENT_NAME=tu_nombre_despliegue

# Configuración de Base de Datos PostgreSQL
POSTGRES_USER=tu_usuario_db
POSTGRES_PASSWORD=tu_contraseña_db
POSTGRES_HOST=tu_host_db
POSTGRES_PORT=5432
POSTGRES_DB=tu_nombre_base_datos
POSTGRES_SSLMODE=prefer
```

### 4.3 Instalación de Dependencias

```bash
git clone <url-del-repositorio>
cd interview-gpt-ai-api
pip install -r requirements.txt
```

### 4.4 Ejecución en Desarrollo

```bash
func start
```

La API estará disponible en `http://localhost:7071`.

---

## Documentación de la API

### 5.1 Endpoints de Gestión de Entrevistas LangGraph

#### `POST /api/interview_chat`
Ejecuta una conversación de entrevista inteligente usando flujo de trabajo LangGraph con validación automatizada de respuestas y preguntas de seguimiento.

**Cuerpo de la Solicitud:**
```json
{
  "thread_id": "string (requerido)",
  "question": {
    "question": "string",
    "context": "string",
    "question_number": 1,
    "total_questions": 5
  },
  "user_data": {
    "user_name": "string"
  },
  "user_response": "string",
  "description": "string (opcional)",
  "language": "string (por defecto: 'es')"
}
```

**Respuesta:**
```json
{
  "status": "success|error",
  "thread_id": "string",
  "is_complete": "boolean|string",
  "validation_result": "string",
  "current_question": {
    "question": "string",
    "context": "string",
    "question_number": 1,
    "total_questions": 5
  },
  "messages": [
    {
      "role": "user|assistant",
      "content": "string"
    }
  ]
}
```

**Características:**
- Flujo de conversación inteligente basado en LangGraph
- Validación de respuestas y verificación de completitud impulsada por IA
- Soporte multiidioma con detección dinámica
- Preguntas de seguimiento conscientes del contexto
- Manejo de filtros de contenido con reformulación automática
- Persistencia de estado de conversación con checkpoints

#### `GET /api/checkpoints`
Recupera checkpoints de entrevista para un hilo específico, permitiendo recuperación de conversación y gestión de estado para entrevistas LangGraph.

**Parámetros de Consulta:**
- `thread_id` (requerido): El identificador del hilo de entrevista

**Respuesta:**
```json
{
  "status": "success|error",
  "thread_id": "string",
  "checkpoints": [
    {
      "id": "string",
      "timestamp": "string",
      "is_complete": "boolean",
      "current_question": {
        "question": "string",
        "context": "string",
        "question_number": 1,
        "total_questions": 5
      },
      "messages": [
        {
          "role": "user|assistant",
          "content": "string"
        }
      ]
    }
  ],
  "last_checkpoint": "object"
}
```

### 5.2 Endpoints de Chat AI General

#### `POST /api/interview-gpt-openai`
Genera texto de IA basado en diferentes tipos de prompts según el caso de uso. Este endpoint es versátil y se utiliza en múltiples escenarios:

**Cuerpo de la Solicitud:**
```json
{
  "prompt": "string (varía según el caso de uso)",
  "temperature": 0.7
}
```

**Casos de Uso:**

1. **Generación de Resultado Final (User Side)**: 
   - Recibe el prompt del administrador concatenado con el historial completo de la entrevista
   - Genera el texto indicado por el admin (reporte, reseña, análisis, etc.) una vez que finaliza la entrevista
   - Utiliza toda la información recopilada durante la conversación

2. **Mejora de Texto Generado (User Side)**:
   - Recibe directamente un input del usuario para mejorar el texto previamente generado
   - Utiliza IA para refinar y mejorar el resultado anterior
   - Permite iteraciones y mejoras continuas del documento

3. **Mejora del Prompt del Admin (Admin Side)**:
   - Recibe el prompt original del administrador
   - Utiliza IA para hacer el prompt más claro, específico y efectivo
   - Ayuda a optimizar las instrucciones para mejores resultados

**Respuesta:** Stream de eventos enviados por servidor con el texto generado o mejorado por IA según el caso de uso específico.

#### `POST /api/chat_ia_interview`
Este endpoint permite a los usuarios hacer preguntas sobre entrevistas completadas y obtener respuestas contextuales basadas en la información de todas las entrevistas proporcionada.

**Cuerpo de la Solicitud:**
```json
{
  "inputUser": "string",
  "systemMessage": "string (opcional)",
  "interviewData": "object",
  "messageHistory": [
    {
      "role": "user|assistant",
      "content": "string"
    }
  ],
  "temperature": 0.7
}
```

**Respuesta:** Stream de eventos enviados por servidor con respuestas de IA contextuales basadas en datos de entrevista.

**Caso de Uso:** 
- Panel de administración para analizar resultados de entrevistas
- Interfaz de chat para discutir hallazgos de entrevistas
- Preguntas y respuestas contextuales sobre respuestas específicas de entrevistas
- Análisis de seguimiento y generación de insights
- Chat contextual que utiliza la información completa de todas las entrevistas para proporcionar respuestas informadas

---

## Arquitectura Técnica

### 6.1 Flujo de Entrevista

El sistema de entrevista implementa un flujo de trabajo basado en LangGraph:

**Nodos:**
1. **validate_response**: Analiza la completitud de la conversación y cobertura del contexto
2. **interviewer**: Genera preguntas de seguimiento inteligentes basadas en el contexto
3. **farewell**: Maneja la finalización de la conversación y gestión de transiciones

**Flujo:**
```
START → validate_response → [condicional] → interviewer/farewell → END
```

**Gestión de Estado:**
- **InterviewState**: Rastrea el estado de la conversación, preguntas y resultados de validación
- **Checkpoints**: Persistencia de PostgreSQL para recuperación de conversación
- **Historial de Mensajes**: Mantenimiento completo del contexto de conversación

### 6.2 Configuración de Base de Datos

El sistema utiliza PostgreSQL con agrupación de conexiones asíncronas:

- **Agrupación de Conexiones**: AsyncConnectionPool con 20 conexiones máximas
- **Checkpointer**: AsyncPostgresSaver para persistencia de estado de LangGraph

### 6.3 Filtrado de Contenido

La API implementa mecanismos de filtrado de contenido en casos donde Azure OpenAI detecta contenido inapropiado incorrectamente:

- **Detección Automática**: Identifica contenido inapropiado usando filtros de Azure OpenAI
- **Reformulación de Mensajes**: Reformula automáticamente contenido problemático que realmente no es inapropiado
- **Adaptación de Prompt del Sistema**: Ajusta las instrucciones del sistema cuando es necesario
- **Lógica de Reintento**: Múltiples intentos con diferentes enfoques para manejo de contenido

---

## Despliegue en Producción

### Despliegue en Azure Functions
```bash
# Construir y desplegar en Azure Functions
func azure functionapp publish <nombre-función-app>
```

### Pruebas de Desarrollo Local
```bash
# Iniciar servidor de desarrollo local
func start

# Probar endpoints
curl -X POST http://localhost:7071/api/interview_chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "test-123", "question": {...}}'
```


---

## Contribuciones

Gracias por tu interés en mejorar Interview GPT AI API.

Actualmente no aceptamos pull requests directos a este repositorio. Sin embargo, si deseas explorar el código, desarrollar nuevas funcionalidades o proponer mejoras, te animamos a crear tu propia versión del proyecto (mediante fork o clonación).

Si tienes sugerencias, detectaste algún error o deseas compartir una propuesta de mejora, no dudes en escribirnos a:  
**techlab@iadb.org**

Estaremos encantados de conocer tus ideas y colaborar en su desarrollo.

---

## Autores

**Alejandra Pérez Ortega**

**Jhoselyn Pajuelo Villanueva**

**Ignacio Cerrato**

**José Daniel Zárate**



---

# Interview GPT AI API

**An AI-powered API service built with Azure Functions, LangGraph, and PostgreSQL that provides intelligent conversation flow management and response validation for the Interview GPT application.**

---

## Table of Contents

1. [What is Interview GPT AI API?](#what-is-interview-gpt-ai-api)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Installation Guide](#installation-guide)
   - [4.1 System Requirements](#41-system-requirements)
   - [4.2 Environment Configuration](#42-environment-configuration)
   - [4.3 Dependencies Installation](#43-dependencies-installation)
   - [4.4 Development Execution](#44-development-execution)
5. [API Documentation](#api-documentation)
   - [5.1 LangGraph Interview Management Endpoints](#51-langgraph-interview-management-endpoints)
   - [5.2 General AI Chat Endpoints](#52-general-ai-chat-endpoints)
6. [Technical Architecture](#technical-architecture)
   - [6.1 Interview Flow](#61-interview-flow)
   - [6.2 Database Configuration](#62-database-configuration)
   - [6.3 Content Filtering](#63-content-filtering)
7. [Production Deployment](#production-deployment)
8. [Contributions](#contributions)
9. [Authors](#authors)

---

## What is Interview GPT AI API?

Interview GPT AI API is an AI-powered service designed to provide intelligent conversational interview capabilities for the Interview GPT application. The system leverages AI capabilities to provide:

- **Intelligent Interview Flow**: Automated interview process with context-aware questioning using LangGraph
- **Response Validation**: AI-powered validation to ensure complete and relevant responses
- **Multi-language Support**: Interviews can be conducted in multiple languages with dynamic language detection
- **Conversation Persistence**: PostgreSQL-based checkpoint system for conversation state management
- **Content Filter Handling**: Automatic content filtering and message rephrasing for appropriate interactions
- **Streaming Responses**: Real-time streaming for AI chat interactions
- **Interview Management**: Complete interview lifecycle management with checkpoints

The platform is built with a microservices architecture using Azure Functions, providing scalability and reliability for production environments.

---

## Key Features

- **Intelligent Conversation Flow**: LangGraph-based workflow with automated follow-up questions
- **Response Validation**: AI-powered completeness assessment and context validation
- **Multi-language Support**: Dynamic language detection and consistent language maintenance
- **Conversation Persistence**: PostgreSQL checkpoints for conversation recovery and state management
- **Content Filter Handling**: Automatic detection and rephrasing of inappropriate content
- **Streaming Responses**: Real-time server-sent events for AI chat interactions
- **Interview Management**: Complete interview lifecycle with checkpoint management
- **Error Handling**: Comprehensive error handling with graceful degradation
- **Monitoring and Logging**: Structured logging and performance monitoring

---

## System Architecture

- **Backend**: Azure Functions with Python for serverless computing
- **AI/ML**: Azure OpenAI with LangGraph for intelligent conversation flow
- **Database**: PostgreSQL with async connection pooling for data persistence
- **Streaming**: Server-sent events for real-time response delivery
- **State Management**: LangGraph checkpoints for conversation state persistence
- **Content Filtering**: Advanced content filtering with automatic rephrasing capabilities

---

## Installation Guide

### 4.1 System Requirements

- **Python** 3.8 or higher
- **Azure Functions Core Tools** for local development
- **PostgreSQL** database (version 12 or higher)
- **Azure OpenAI** service with appropriate API access
- **Git** 2.34 or higher (for repository cloning)

### 4.2 Environment Configuration

Create a `.env` file in the project root with the following variables:

```env
# Azure OpenAI Configuration
AZURE_OPEN_AI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_API_INSTANCE_NAME=your_instance_name
AZURE_OPENAI_API_BASE_PATH=your_base_path
AZURE_DEPLOYMENT_NAME=your_deployment_name

# PostgreSQL Database Configuration
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password
POSTGRES_HOST=your_db_host
POSTGRES_PORT=5432
POSTGRES_DB=your_database_name
POSTGRES_SSLMODE=prefer
```

### 4.3 Dependencies Installation

```bash
git clone <repository-url>
cd interview-gpt-ai-api
pip install -r requirements.txt
```

### 4.4 Development Execution

```bash
func start
```

The API will be available at `http://localhost:7071`.

---

## API Documentation

### 5.1 LangGraph Interview Management Endpoints

#### `POST /api/interview_chat`
Executes an intelligent interview conversation using LangGraph workflow with automated response validation and follow-up questions.

**Request Body:**
```json
{
  "thread_id": "string (required)",
  "question": {
    "question": "string",
    "context": "string",
    "question_number": 1,
    "total_questions": 5
  },
  "user_data": {
    "user_name": "string"
  },
  "user_response": "string",
  "description": "string (optional)",
  "language": "string (default: 'es')"
}
```

**Response:**
```json
{
  "status": "success|error",
  "thread_id": "string",
  "is_complete": "boolean|string",
  "validation_result": "string",
  "current_question": {
    "question": "string",
    "context": "string",
    "question_number": 1,
    "total_questions": 5
  },
  "messages": [
    {
      "role": "user|assistant",
      "content": "string"
    }
  ]
}
```

**Features:**
- LangGraph-based intelligent conversation flow
- AI-powered response validation and completion checking
- Multi-language support with dynamic detection
- Context-aware follow-up questions
- Content filter handling with automatic rephrasing
- Conversation state persistence with checkpoints

#### `GET /api/checkpoints`
Retrieves interview checkpoints for a specific thread, enabling conversation recovery and state management for LangGraph interviews.

**Query Parameters:**
- `thread_id` (required): The interview thread identifier

**Response:**
```json
{
  "status": "success|error",
  "thread_id": "string",
  "checkpoints": [
    {
      "id": "string",
      "timestamp": "string",
      "is_complete": "boolean",
      "current_question": {
        "question": "string",
        "context": "string",
        "question_number": 1,
        "total_questions": 5
      },
      "messages": [
        {
          "role": "user|assistant",
          "content": "string"
        }
      ]
    }
  ],
  "last_checkpoint": "object"
}
```

### 5.2 General AI Chat Endpoints

#### `POST /api/interview-gpt-openai`
Generates AI text based on different types of prompts according to the use case. This endpoint is versatile and is used in multiple scenarios:

**Request Body:**
```json
{
  "prompt": "string (varies according to use case)",
  "temperature": 0.7
}
```

**Use Cases:**

1. **Final Result Generation (User Side)**: 
   - Receives the administrator's prompt concatenated with the complete interview history
   - Generates the text indicated by the admin (report, review, analysis, etc.) once the interview is completed
   - Uses all information collected during the conversation

2. **Generated Text Improvement (User Side)**:
   - Receives direct input from the user to improve previously generated text
   - Uses AI to refine and improve the previous result
   - Allows continuous iterations and improvements of the document

3. **Admin Prompt Improvement (Admin Side)**:
   - Receives the original administrator's prompt
   - Uses AI to make the prompt clearer, more specific and effective
   - Helps optimize instructions for better results

**Response:** Server-sent events stream with AI-generated or improved text according to the specific use case.

#### `POST /api/chat_ia_interview`
Enables AI-powered chat about interview data with context awareness. This endpoint allows users to ask questions about completed interviews and get contextual responses.

**Request Body:**
```json
{
  "inputUser": "string",
  "systemMessage": "string (opcional)",
  "interviewData": "object",
  "messageHistory": [
    {
      "role": "user|assistant",
      "content": "string"
    }
  ],
  "temperature": 0.7
}
```

**Response:** Server-sent events stream with contextual AI responses based on interview data.

**Use Case:** 
- Admin panel for analyzing interview results
- Chat interface for discussing interview findings
- Contextual Q&A about specific interview responses
- Follow-up analysis and insights generation
- Chat contextual that utilizes the complete information of all interviews to provide informed responses

---

## Technical Architecture

### 6.1 Interview Flow

The interview system implements a LangGraph-based workflow:

**Nodes:**
1. **validate_response**: Analyzes conversation completeness and context coverage
2. **interviewer**: Generates intelligent follow-up questions based on context
3. **farewell**: Handles conversation completion and transition management

**Flow:**
```
START → validate_response → [conditional] → interviewer/farewell → END
```

**State Management:**
- **InterviewState**: Tracks conversation state, questions, and validation results
- **Checkpoints**: PostgreSQL persistence for conversation recovery
- **Message History**: Complete conversation context maintenance

### 6.2 Database Configuration

The system utilizes PostgreSQL with async connection pooling:

- **Connection Pool**: AsyncConnectionPool with 20 maximum connections
- **Checkpointer**: AsyncPostgresSaver for LangGraph state persistence
- **Row Factory**: dict_row for simplified data access
- **SSL Mode**: Configurable SSL connection settings for security

### 6.3 Content Filtering

The API implements content filtering mechanisms where the Azure OpenAI API detect an innapropiate content incorrectly:

- **Automatic Detection**: Identifies inappropriate content using Azure OpenAI filters
- **Message Rephrasing**: Automatically reformulates problematic content that is not really innapropiate
- **System Prompt Adaptation**: Adjusts system instructions when needed
- **Retry Logic**: Multiple attempts with different approaches for content handling

---

## Production Deployment

### Azure Functions Deployment
```bash
# Build and deploy to Azure Functions
func azure functionapp publish <function-app-name>
```

### Local Development Testing
```bash
# Start local development server
func start

# Test endpoints
curl -X POST http://localhost:7071/api/interview_chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "test-123", "question": {...}}'
```

### Monitoring and Logging
- **Structured Logging**: JSON-formatted logs for easy parsing and analysis
- **Performance Metrics**: Response time and throughput monitoring
- **Error Tracking**: Detailed error logging with context information
- **Health Checks**: Endpoint availability monitoring for production environments

---

## Contributions

Thank you for your interest in improving Interview GPT AI API.

Currently, we do not accept direct pull requests to this repository. However, if you wish to explore the code, develop new features, or propose improvements, we encourage you to create your own version of the project (through fork or cloning).

If you have suggestions, detect any errors, or wish to share improvement proposals, please do not hesitate to contact us at:  
**techlab@iadb.org**

We will be delighted to learn about your ideas and collaborate on their development.

---

## Authors

**Alejandra Pérez Ortega**

**Jhoselyn Pajuelo Villanueva**

**Ignacio Cerrato**

**José Daniel Zárate**
