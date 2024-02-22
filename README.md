# SRI_Plagio
## Proyecto de SRI que identifica si existe plagio dado 2 documentos
### Autores:
- Ovidio Navarro Pazos
- Juan José Muñoz Noda
- Jesus Armando Padrón

### Descripción del problema
Detectar plagio entre dos documentos , en caso de que exista, señalar la sección dentro del texto donde se evidencie

### Consideraciones tomadas:
- Idioma de los textos: español
- tamano de n-gramas: 10
- Umbral: 0.7
### Solución desarrolada:  
Se procesan los documento de la carpeta ``data``,estos documentos pueden ser tipo txt o pdf, cada documento se guarda en una lista, se aplica la tokenización utilizando ``spacy``(esp), se elimina el ruido de los documentos como caracteres especiales y las stop-words.  
Utilizando ``Word2Vec`` se halla el embedding (representación semántica por contexto de cada palabra dentro de un documento)  de los documentos, lo cual devuelve un modelo. Después de ralizar el embedding de cada documento, cada palabra esta asociada a un vector del tamaño definido en los argumentos del Word2Vec. Se halla el vector representativo de un documento calculando el promedio de los vectores de cada palabras de ese documento. Para verificar el plagio se utiliza la distancia del coseno con los vectores representativos de los documentos escogidos.  
Se escogió 0.7 como umbral para detectar si existe plagio.
Para detectar las partes del documento donde existe el plagio se utilzan n-gramas de tamaño 10 los cuales se convierten en sets y se hallan las intersecciones (el tamaño de los n-gramas se puede modificar a conveniencia)  
### Ejecutar:
- Instalar depencias que se encuentran en ``requirements.txt``  
- Ejecutar ``start.py`` de el directorio src  
### Trabajo futuro:
- Posibilidad importar nuevos documentos
- Poder selecionar el idioma
- Escoger n-grama a conveniencia 
- Selecionar umbral deseado

