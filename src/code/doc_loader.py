
def_path = "../../data/"

def load_doc(name):
    # Abre el archivo en modo lectura ('r')
    with open(def_path + name, 'r') as archivo:
        # Lee el contenido del archivo
        doc = archivo.read()

    # Imprime el contenido del archivo
    return doc

doc1 = load_doc("naturaleza.txt")
doc2 = load_doc("naturaleza_plagio.txt")

print (doc2)




