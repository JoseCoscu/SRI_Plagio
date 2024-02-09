import tkinter as tk
from tkinter import filedialog


def cargar_archivo(entrada_texto, resaltar_texto):
    archivo = filedialog.askopenfilename(filetypes=[("Archivos de texto", "*.txt")])
    with open(archivo, 'r') as f:
        contenido = f.read()
        entrada_texto.delete(1.0, tk.END)  # Limpiar el contenido anterior
        entrada_texto.insert(tk.END, contenido)  # Mostrar el contenido en el widget

        # Resaltar el texto específico
        indice_inicio = entrada_texto.search(resaltar_texto, "1.0", tk.END)
        while indice_inicio:
            indice_fin = f"{indice_inicio}+{len(resaltar_texto)}c"
            entrada_texto.tag_add("resaltado", indice_inicio, indice_fin)
            indice_inicio = entrada_texto.search(resaltar_texto, indice_fin, tk.END)


# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Visor de archivos de texto")

# Crear frames para dividir la ventana en dos mitades
frame_izquierda = tk.Frame(ventana)
frame_izquierda.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)

frame_derecha = tk.Frame(ventana)
frame_derecha.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5, pady=5)

# Crear widgets de texto para mostrar el contenido de los archivos
texto_izquierda = tk.Text(frame_izquierda)
texto_izquierda.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

texto_derecha = tk.Text(frame_derecha)
texto_derecha.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

# Definir una etiqueta para resaltar el texto
texto_izquierda.tag_configure("resaltado", background="yellow")

# Botones para cargar archivos en la parte inferior y centrada del frame izquierdo
boton_izquierda = tk.Button(frame_izquierda, text="Cargar archivo",
                            command=lambda: cargar_archivo(texto_izquierda, "la vida se adapta a condiciones extremas"))
boton_izquierda.pack(side=tk.BOTTOM)

# Botones para cargar archivos en la parte inferior y centrada del frame derecho
boton_derecha = tk.Button(frame_derecha, text="Cargar archivo",
                          command=lambda: cargar_archivo(texto_derecha, "texto a resaltar"))
boton_derecha.pack(side=tk.BOTTOM)

# Ejecutar la aplicación
ventana.mainloop()
