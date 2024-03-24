from flask import Flask
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import geopy.distance
import io
import base64
from flask import session, Response
from geopy.distance import geodesic


app = Flask(__name__)
app.secret_key = 'mmmbs'
# Dossier pour stocker les fichiers téléchargés
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extraire_coordonnees_de_excel(df):
    """
    Extrait les coordonnées des villes à partir d'un DataFrame.

    Args:
    df (pandas.DataFrame): DataFrame contenant les données des coordonnées des villes.

    Returns:
    dict: Dictionnaire où les clés sont les noms des villes et les valeurs sont les coordonnées (latitude, longitude).
    """
    coords = {}
    for index, row in df.iterrows():
        ville = row['Ville']
        latitude = row['Latitude']
        longitude = row['Longitude']
        coords[ville] = (longitude, latitude)
    return coords


def create_graph_from_data(coords):
    """
    Crée un graphe non orienté pondéré à partir des coordonnées des villes.

    Args:
    coords (dict): Dictionnaire où les clés sont les noms des villes et les valeurs sont les coordonnées (latitude, longitude).

    Returns:
    nx.Graph: Graphe non orienté pondéré avec les villes comme sommets et les distances comme poids sur les arêtes.
    """
    G = nx.Graph()

    # Ajouter les villes comme sommets avec les coordonnées GPS
    for nom, coord in coords.items():
        G.add_node(nom, pos=coord)

    # Calculer les distances entre chaque paire de villes et ajouter des arêtes
    for u in coords:
        for v in coords:
            if u != v:
                distance_uv = geopy.distance.geodesic(coords[u], coords[v]).km
                G.add_edge(u, v, weight=distance_uv)
    return G

def ant_tour(graph, pheromones, alpha, beta):
    tour = [random.choice(list(graph.nodes))]
    while len(tour) < graph.number_of_nodes():
        current_node = tour[-1]
        unvisited = [node for node in graph.nodes() if node not in tour]
        probabilities = []
        total = 0
        for unvisited_node in unvisited:
            pheromone = pheromones[current_node][unvisited_node]
            distance = graph[current_node][unvisited_node]['weight']
            if distance == 0:  # Handle division by zero
                distance = 0.0001  # Small constant added to avoid division by zero
            total += (pheromone ** alpha) * ((1.0 / distance) ** beta)
        for unvisited_node in unvisited:
            pheromone = pheromones[current_node][unvisited_node]
            distance = graph[current_node][unvisited_node]['weight']
            if distance == 0:  # Handle division by zero
                distance = 0.0001  # Small constant added to avoid division by zero
            probability = ((pheromone ** alpha) * ((1.0 / distance) ** beta)) / total
            probabilities.append(probability)
        
        # Handle the case when total is zero after adding small constant
        if total == 0:
            probabilities = [1 / len(unvisited) for _ in unvisited]  # Equal probabilities for unvisited nodes
        
        next_node = random.choices(unvisited, probabilities)[0]
        tour.append(next_node)
    return tour

def update_pheromones(graph, pheromones, tours, evaporation_rate, pheromone_deposit):
    # Ensure all nodes are initialized in the pheromones dictionary
    for node1 in pheromones:
        for node2 in pheromones[node1]:
            if (node1, node2) not in graph.edges() and (node2, node1) not in graph.edges():
                pheromones[node1][node2] = 0.01

    for tour in tours:
        for i in range(len(tour)):
            current_node = tour[i]
            next_node = tour[(i + 1) % len(tour)]  # Circular tour
            pheromones[current_node][next_node] += pheromone_deposit
            pheromones[next_node][current_node] += pheromone_deposit

    # Evaporation
    for node1 in pheromones:
        for node2 in pheromones[node1]:
            pheromones[node1][node2] *= (1.0 - evaporation_rate)
            pheromones[node1][node2] = max(pheromones[node1][node2], 0.01)  # Limit minimum value

@app.route('/result')
def index():
    return render_template('index.html')

@app.route('/')
def animation():
    return render_template('animation.html')

from flask import session
@app.route('/upload', methods=['GET', 'POST'])
def upload_and_choose_city():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Lecture des données du fichier sans l'enregistrer
            data = pd.read_excel(file)
            # Convertir le DataFrame en dictionnaire
            data_dict = data.to_dict(orient='records')
            # Stockage des données dans la session
            session['data'] = data_dict
            
            city = request.form['city']
            algorithm = request.form['algorithm']
            if algorithm == 'ant':
                return redirect(url_for('run_ant_algorithm', city=city))
            else :
                return redirect(url_for('run_tsp_algorithm', city=city))
    return render_template('index.html')

@app.route('/run_ant_algorithm/<city>')
def run_ant_algorithm(city):
    data = session.get('data')
    if data is None:
        flash('No data available')
        return redirect(url_for('index'))
    # Convertir le dictionnaire en DataFrame
    data_df = pd.DataFrame(data)
    # Extraire les coordonnées des villes et créer le graphe à partir des données
    coords = extraire_coordonnees_de_excel(data_df)
    G = create_graph_from_data(coords)

    if city not in G.nodes:
        flash("Ville de départ invalide.")
        return redirect(url_for('index'))
    
    # Paramètres pour l'algorithme ACO
    num_ants = 10
    alpha = 1.0
    beta = 2.0
    evaporation_rate = 0.5
    pheromone_deposit = 1.0
    iterations = 100

    # Initialisation des phéromones
    pheromones = {node1: {node2: 1.0 for node2 in G.nodes()} for node1 in G.nodes()}

    best_tour = None
    best_distance = float('inf')
    for _ in range(iterations):
        ant_tours = [ant_tour(G, pheromones, alpha, beta) for _ in range(num_ants)]
        update_pheromones(G, pheromones, ant_tours, evaporation_rate, pheromone_deposit)
        # Recherche du meilleur parcours de cette itération
        for tour in ant_tours:
            tour_distance = sum([G[tour[i]][tour[i + 1]]['weight'] for i in range(len(tour) - 1)])
            tour_distance += G[tour[-1]][tour[0]]['weight']  # Retour à la première ville
            if tour_distance < best_distance:
                best_distance = tour_distance
                best_tour = tour

    # Gestion de 'nan' (si cela représente Zoueratt répété)
    best_tour = [city if city != 'nan' else 'Zoueratt' for city in best_tour]

    # Recherche de l'index de la ville choisie dans le parcours
    city_index = best_tour.index(city)
    # Rotation du parcours pour commencer par la ville choisie
    best_tour = best_tour[city_index:] + best_tour[:city_index]

    # Calcul et affichage des distances entre les villes
    distances = []
    for i in range(len(best_tour) - 1):
        city1 = best_tour[i]
        city2 = best_tour[i + 1]
        distance = G[city1][city2]['weight']
        distances.append((city1, city2, distance))

    # Créer le graphe à afficher dans la page HTML
    fig, ax = plt.subplots()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray', linewidths=1, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=[(best_tour[i], best_tour[i + 1]) for i in range(len(best_tour) - 1)] + [(best_tour[-1], best_tour[0])], edge_color='g', width=3, ax=ax)
    ax.set_title("Meilleur parcours en utilisant ACO ")

    # Convertir l'image en base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    # Retourner l'image base64 et d'autres données à afficher dans la page HTML
    return render_template('index.html', best_tour=best_tour, best_distance=best_distance, distances=distances, image_base64=image_base64)

# --------------------------------------------------------------------------
# Vos fonctions existantes

def charger_graphe(nom_fichier):
    """Charger les données à partir du fichier Excel et créer un graphe non orienté et pondéré."""
    data = pd.read_excel(nom_fichier)
    coords = extraire_coordonnees_de_excel(data)
    G = create_graph_from_data(coords)
    return G

def distance(ville1, ville2, G):
    """Calculer la distance entre deux villes en utilisant la formule de geodesic."""
    pos1 = G.nodes[ville1]['pos']
    pos2 = G.nodes[ville2]['pos']
    return geodesic(pos1, pos2).kilometers

def trouver_chemin_optimal(G, ville_depart):
    """Trouver le chemin optimal à partir de la ville de départ."""
    mst = nx.minimum_spanning_tree(G)
    dfs_path = list(nx.dfs_preorder_nodes(mst))
    if ville_depart not in dfs_path:
        print(f"{ville_depart} n'est pas incluse dans l'arbre couvrant minimal.")
        return None
    start_index = dfs_path.index(ville_depart)
    dfs_path = dfs_path[start_index:] + dfs_path[:start_index]
    return dfs_path

import base64
import matplotlib.pyplot as plt
import io

def charger_graphe(nom_fichier):
    """Charger les données à partir du fichier Excel et créer un graphe non orienté et pondéré."""
    data = pd.read_excel(nom_fichier)
    coords = extraire_coordonnees_de_excel(data)
    G = create_graph_from_data(coords)
    return G

def distance(ville1, ville2, G):
    """Calculer la distance entre deux villes en utilisant la formule de geodesic."""
    pos1 = G.nodes[ville1]['pos']
    pos2 = G.nodes[ville2]['pos']
    return geodesic(pos1, pos2).kilometers

def trouver_chemin_optimal(G, ville_depart):
    """Trouver le chemin optimal à partir de la ville de départ."""
    mst = nx.minimum_spanning_tree(G)
    dfs_path = list(nx.dfs_preorder_nodes(mst))
    if ville_depart not in dfs_path:
        print(f"{ville_depart} n'est pas incluse dans l'arbre couvrant minimal.")
        return None
    start_index = dfs_path.index(ville_depart)
    dfs_path = dfs_path[start_index:] + dfs_path[:start_index]
    return dfs_path


def afficher_graphe(G, tour):
    """Afficher le graphe et le chemin optimal."""
    plt.figure(figsize=(9, 7))
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=False, node_size=300, node_color='lightblue', font_size=8)
    edges = [(tour[i], tour[i+1]) for i in range(len(tour)-1)]
    labels = {(tour[i], tour[i+1]): f"{distance(tour[i], tour[i+1], G):.2f} km" for i in range(len(tour)-1)}
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, edge_color='red')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='blue')
    for i, city in enumerate(tour[:-1]):
        x, y = pos[city]
        plt.text(x + 0.05, y + 0.05, city, fontsize=8, ha='right', color='black')
        plt.text(x + 0.1, y + 0.1, str(i + 1), fontsize=8, ha='right', color='red')
    plt.title("Tournée en Mauritanie avec distance minimale")

    # Convertir l'image en base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    # Afficher l'image dans la page HTML
    return image_base64


def tsp_algorithm(G, city):
    tour = trouver_chemin_optimal(G, city)
    if tour:
        flash("Chemin le plus court :")
        for i in range(len(tour)-1):
            flash(f"{i+1}. {tour[i]} - {tour[i+1]} : {distance(tour[i], tour[i+1], G):.2f} km")
        total_distance = sum(distance(tour[i], tour[i+1], G) for i in range(len(tour)-1))
        flash(f"Distance totale : {total_distance:.2f} km")
        image_base64 = afficher_graphe(G, tour)


        #Retourner l'image base64 et d'autres données à afficher dans la page HTML
        return render_template('index2.html', image_base64=image_base64)
    else:
        flash("Impossible de trouver un chemin optimal.")
        return redirect(url_for('index'))

@app.route('/run_tsp_algorithm/<city>')
def run_tsp_algorithm(city):
    data = session.get('data')
    if data is None:
        flash('No data available')
        return redirect(url_for('index'))
    
    # Convertir le dictionnaire en DataFrame
    data_df = pd.DataFrame(data)
    
    # Extraire les coordonnées des villes et créer le graphe à partir des données
    coords = extraire_coordonnees_de_excel(data_df)
    G = create_graph_from_data(coords)
    
    if city not in G.nodes:
        flash("Ville de départ invalide.")
        return redirect(url_for('index'))
    
    return tsp_algorithm(G, city)


if __name__ == "__main__":
    app.run(debug=True)


