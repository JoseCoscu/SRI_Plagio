import tkinter as tk
from tkinter import filedialog


def cargar_archivo(entrada_texto, resaltar_textos, doc=False):
    if not doc:
        archivo = filedialog.askopenfilename(filetypes=[("Archivos de texto", "*.txt")])
        with open(archivo, 'r', encoding='utf-8') as f:
            contenido = f.read()
            entrada_texto.delete(1.0, tk.END)  # Limpiar el contenido anterior
            entrada_texto.insert(tk.END, contenido)  # Mostrar el contenido en el widget
    else:
        entrada_texto.delete(1.0, tk.END)  # Limpiar el contenido anterior
        entrada_texto.insert(tk.END, doc)  # Mostrar el contenido en el widget

        # Resaltar los textos específicos
    for resaltar_texto in resaltar_textos:
        indice_inicio = "1.0"
        while indice_inicio:
            indice_inicio = entrada_texto.search(resaltar_texto, indice_inicio, tk.END)
            if indice_inicio:
                indice_fin = f"{indice_inicio}+{len(resaltar_texto)}c"
                entrada_texto.tag_add("resaltado", indice_inicio, indice_fin)
                indice_inicio = indice_fin


def create_window():
    # Crear la ventana principal
    ventana = tk.Tk()
    ventana.title("Visor de archivos de texto")
    return ventana
