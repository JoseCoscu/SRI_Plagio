# SRI_Plagio
## Proyecto de SRI que identifica si existe plagio dado 2 documentos
### Algoritmo:  
Se procesan los documento de la carpeta ``data``,estos documentos pueden ser tipo txt o pdf, cada documento se guarda en una lista, se aplica la tokenizacion utilizando ``spacy``(esp),se elimina el ruido de los documentos como caracteres especiales y las stop-words.  
Utilizando ``Word2Vec`` hallamos el embedding (representacion semantica por contexto de cada palabra dentro de un documento)  de los documentos,lo cual devuelve un modelo. Despues de ralizar el embedding de cada documento cada palabra esta asociada a un vector del tamaño definido en los argumentos del Word2Vec.Hallamos un vector representativo de un documento calculando el promedio de los vectores de cada palabras de ese documento. Para verifocar el plagio utilizamos la distancia del coseno con los vectores representativos de los documentos escogidos.  
Escogimos 0.7 como umbral para detectar si existe plagio.
Para detectar las partes del documento donde existe el plagio utilzamos n-gramas de tamaño 10 los cuales se convierten en sets y se hallan las intersecciones (el tamaño de los n-gramas se puede modificar a conveniencia)  
### Ejecutar:
- Instalar depencias que se encuentran en ``requirements.txt``  
- Ejecutar ``start.py`` de el directorio src  
### Trabajo futuro:
- Poder importar nuevos documentos
- Poder selecionar el idioma
- Escoger n-grama a conveniencia 
- Selecionar umbral deseado

