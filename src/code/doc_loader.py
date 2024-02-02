
def_path = "../../data/"

def load_doc(name):
    # Abre el archivo en modo lectura ('r')
    with open(name, 'r') as archivo:
        # Lee el contenido del archivo
        doc = archivo.read()

    # Imprime el contenido del archivo
    return doc

doc1 = load_doc(def_path + "naturaleza.txt")

print (doc1)




