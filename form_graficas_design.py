import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FormularioGraficasDesign:

    def __init__(self, panel_principal):
        # Crear una figura y los subgráficos
        figura = Figure(figsize=(14, 12), dpi=100)
        ax2 = figura.add_subplot(121, projection='polar')  # Gráfico polar
        ax4 = figura.add_subplot(122)  # Gráfico de cebolla

        # Cargar el DataFrame
        self.df_similarities = pd.read_csv("resultados_clasificacion.csv")

        # Graficar en los subgráficos
        self.grafico2(ax2)
        self.grafico4(ax4)

        # Agregar los gráficos a la ventana de Tkinter
        canvas = FigureCanvasTkAgg(figura, master=panel_principal)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def grafico2(self, ax):
        # Gráfico polar
        if 'algoritmo_inseguro' in self.df_similarities.columns:
            insecure_counts = self.df_similarities['algoritmo_inseguro'].value_counts()
            categories = insecure_counts.index
            values = insecure_counts.values

            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # Cerrar el gráfico
            values = np.concatenate((values, [values[0]]))  # Cerrar el gráfico

            ax.bar(angles[:-1], values[:-1], color=['#DB1459', '#36545C', '#35879C', '#ffcc99'], width=0.3, align='edge')

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_title('Cantidad de Imágenes Inseguras por Algoritmo', pad=20)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
        else:
            ax.text(0.5, 0.5, 'Columna "algoritmo_inseguro" no encontrada', 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='red')

    def grafico4(self, ax):
        # Gráfico de cebolla
        if 'distancia_juniward' in self.df_similarities.columns and \
           'distancia_uerd' in self.df_similarities.columns and \
           'distancia_jmipod' in self.df_similarities.columns:
            distances = {
                'Juniward': self.df_similarities['distancia_juniward'].values[10],
                'UERD': self.df_similarities['distancia_uerd'].values[10],
                'JMiPOD': self.df_similarities['distancia_jmipod'].values[10]
            }

            sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1], reverse=False))

            colors = ['#35879C', '#36545C', '#DB1459', '#ffcc99']

            radii = []
            for i, (algorithm, distance) in enumerate(sorted_distances.items()):
                outer_radius = distance * 100
                radii.append(outer_radius)
                color = colors[i % len(colors)]
                wedge = patches.Wedge(
                    center=(0, 0),
                    r=outer_radius,
                    theta1=0,
                    theta2=360,
                    width=outer_radius - (radii[i - 1] if i > 0 else 0),
                    facecolor=color,
                    edgecolor=None  # Eliminar el borde
                )
                ax.add_patch(wedge)

                mid_radius = (radii[i - 1] if i > 0 else 0) + (outer_radius - (radii[i - 1] if i > 0 else 0)) / 2
                ax.text(0, mid_radius, algorithm, horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')

            ax.set_xlim(-max(radii) * 1.2, max(radii) * 1.2)
            ax.set_ylim(-max(radii) * 1.2, max(radii) * 1.2)
            ax.set_aspect('equal', 'box')
            ax.set_title('Diagrama de Cebolla - Distancias de Algoritmos', size=15, color='black')

            # Eliminar los ejes y el marco
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Faltan datos para el gráfico de cebolla', 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='red')

# Crear una ventana Tkinter
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Formulario de Gráficas")

    # Instanciar la clase FormularioGraficasDesign
    app = FormularioGraficasDesign(root)

    # Ejecutar la aplicación Tkinter
    root.mainloop()
