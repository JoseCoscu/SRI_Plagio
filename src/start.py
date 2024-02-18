from code2 import doc_loader as dl, plagio_checker as pc
from gui import view as v
import tkinter as tk
from tkinter import messagebox
import time


def compare_docs(vcts):
    contenido_izquierda = texto_izquierda.get("1.0", tk.END)
    contenido_derecha = texto_derecha.get("1.0", tk.END)

    if (len(contenido_derecha) == 1 or len(contenido_izquierda) == 1):
        messagebox.showwarning("Warning", "Nothig to compare!")
    index1, index2, docs = get_doc_index(contenido_izquierda, contenido_derecha)
    v_sim = dl.v_similarity(vcts[index1], vcts[index2])

    if (v_sim >= 0.8):
        messagebox.showwarning(f"Warning", f"Plagarism Detecteddd!!!\n{"{:.2f}".format(v_sim * 100)}% Precision")
        conmon_ngrams = pc.find_similar_ngrams(docs[index1], docs[index2])
        v.cargar_archivo(texto_izquierda, conmon_ngrams, docs[index1])
      #  time.sleep(2)
        v.cargar_archivo(texto_derecha, conmon_ngrams, docs[index2])


    else:
        messagebox.showwarning("Warning", "Nothig detected!")


def get_doc_index(doc1, doc2):
    doc1 = doc1.lower().split()
    doc2 = doc2.lower().split()
    docs = dl.load_docs(False)
    index1 = 0
    index2 = 0
    for i in range(len(docs)):
        if (doc1 == docs[i].lower().split()):
            index1 = i
        if (doc2 == docs[i].lower().split()):
            index2 = i

    return index1, index2, docs


vcts = pc.process_docs()

ventana = v.create_window()

# Crear frame principal que ocupa toda la ventana
frame_principal = tk.Frame(ventana)
frame_principal.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=5, pady=5)

# Crear frames para dividir la ventana en dos mitades
frame_izquierda = tk.Frame(frame_principal)
frame_izquierda.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)

frame_derecha = tk.Frame(frame_principal)
frame_derecha.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5, pady=5)

# Crear widgets de texto para mostrar el contenido de los archivos
texto_izquierda = tk.Text(frame_izquierda)
texto_izquierda.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

texto_derecha = tk.Text(frame_derecha)
texto_derecha.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

# Definir una etiqueta para resaltar el texto
texto_izquierda.tag_configure("resaltado", background="yellow")
texto_derecha.tag_configure("resaltado", background="yellow")

# Botones para cargar archivos en la parte inferior y centrada del frame izquierdo
boton_izquierda = tk.Button(frame_izquierda, text="Load File",
                            command=lambda: v.cargar_archivo(texto_izquierda,''))

boton_izquierda.pack(side=tk.BOTTOM)

# Botones para cargar archivos en la parte inferior y centrada del frame derecho
boton_derecha = tk.Button(frame_derecha, text="Load File",
                          command=lambda: v.cargar_archivo(texto_derecha, ""))
boton_derecha.pack(side=tk.BOTTOM)

# Crear un botón en el centro del frame principal
boton_central = tk.Button(frame_principal, text="Verify", command=lambda: compare_docs(vcts))
from tkinter import filedialog

boton_central.pack(side=tk.BOTTOM, pady=10)

boton_cerrar = tk.Button(frame_principal, text="Cerrar Ventana", command=ventana.destroy)
boton_cerrar.pack()

# Ejecutar la aplicación
ventana.mainloop()