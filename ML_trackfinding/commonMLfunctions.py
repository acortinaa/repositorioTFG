from common import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from model import TripletNet, SimpleNet

def filtering_by_theta(hits):
    # 1. Primer volumen
    first_volume = hits.volume_id.unique()
    first_volume = sorted(first_volume)[0]

    hits_vol = hits[hits.volume_id == first_volume].copy()

    # 2. Ordenamos por valor absoluto de z
    hits_vol['abs_z'] = hits_vol['z'].abs()
    hits_vol_sorted = hits_vol.sort_values(by='abs_z')

    # 3. Tomamos el mínimo y máximo z en valor absoluto
    min_z_hit = hits_vol_sorted.iloc[0]
    max_z_hit = hits_vol_sorted.iloc[-1]

    # 4. Calculamos el ángulo theta en grados para ambos hits
    def compute_theta_deg(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        # Evitar división por cero
        if r == 0:
            return 0.0
        theta = np.degrees(np.arccos(z / r))
        return theta

    theta_min = compute_theta_deg(min_z_hit['x'], min_z_hit['y'], min_z_hit['z'])
    theta_max = compute_theta_deg(max_z_hit['x'], max_z_hit['y'], max_z_hit['z'])

    print(f"Ángulo theta mínimo (z más pequeño en valor absoluto): {theta_min:.2f}°")
    print(f"Ángulo theta máximo (z más grande en valor absoluto): {theta_max:.2f}°")
    return theta_min, theta_max


def plot_all_positive_triplets(triplets, labels, layer1, layer2, layer3):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar todos los hits de las 3 capas en gris con baja opacidad
    ax.scatter(layer1.x, layer1.y, layer1.z, c='gray', s=2, alpha=0.3, label='Layer 1')
    ax.scatter(layer2.x, layer2.y, layer2.z, c='gray', s=2, alpha=0.3, label='Layer 2')
    ax.scatter(layer3.x, layer3.y, layer3.z, c='gray', s=2, alpha=0.3, label='Layer 3')

    # Filtrar solo tripletes con label 1 (positivos)
    pos_idx = np.where(labels == 1)[0]

    # Para no saturar el plot, limita a máximo N tripletes
    N = 50  
    count = 0

    for idx in pos_idx:
        triplet = triplets[idx].reshape(3, 3)  # (3 hits, xyz)
        # Conectar los 3 puntos con línea
        ax.plot(triplet[:, 0], triplet[:, 1], triplet[:, 2], color='blue', alpha=0.7)
        # Poner los puntos del triplete
        ax.scatter(triplet[:, 0], triplet[:, 1], triplet[:, 2], c='blue', s=20, alpha=0.9)
        count += 1
        if count >= N:
            break

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Tripletes con Label=1 (mostrando {count} de {len(pos_idx)})')
    ax.legend()
    plt.show()


# Dataset personalizado
import torch
from torch.utils.data import Dataset
import numpy as np

class TripletDataset(Dataset):
    def __init__(self, X, y):
        self.X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Dataset y dataloaders
from torch.utils.data import DataLoader, random_split

def create_dataloaders(X, y, batch_size, seed=42,
                       num_workers=0, pin_memory=False):
    dataset = TripletDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    generator  = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    print(f"Dataset length: {len(dataset)}, train: {len(train_ds)}, val: {len(val_ds)}")
    return train_loader, val_loader


def hits_vertex(hits, particles, truth, PARTICLES_FROM_VERTEX):
    ''' Filtra los hits para quedarnos solo con los del detector central '''

    def distance(particle):
        ''' Distancia en mm de la partícula al origen'''
        return np.sqrt(particle.vx**2 + particle.vy**2 + particle.vz**2)

    particles['r'] = distance(particles)
    particles['phi'] = np.arctan2(particles.vy, particles.vx)
    particles['theta'] = np.arccos(particles.vz / particles.r)
    particles['p'] = np.sqrt(particles.px**2 + particles.py**2 + particles.pz**2)
    particles['pt'] = np.sqrt(particles.px**2 + particles.py**2) 
    
    truth = truth[truth.particle_id.isin(particles.particle_id)]
    particles_all = particles

    if PARTICLES_FROM_VERTEX:
        # Voy a coger solo las partículas con r < 2.6
        particles = particles[particles.r < 2.6]

        # Con ese radio, voy a coger solo las partículas con z entre -25 y 25 mm
        particles = particles[(particles.vz > -25) & (particles.vz < 25)]

        # Del truth cojo solo las partículas que están en particles
        truth = truth[truth.particle_id.isin(particles.particle_id)]

        # Cojo ahora los hits_id que están en truth
        hits_all = hits
        hits = hits[hits.hit_id.isin(truth.hit_id)]

        print("Los datos que tomo son un {:.4f}% de los datos originales".format(hits.shape[0]/hits_all.shape[0]*100))

    # Unimos los particle_id a hits
    hits = hits.merge(truth[['hit_id', 'particle_id']], on='hit_id', how='left')

    hits['phi'] = np.degrees(np.arctan2(hits['y'], hits['x']))
    hits['phi'] = (hits['phi'] + 360) % 360  # Normaliza a [0, 360)
    hits['theta'] = np.degrees(np.arctan2(np.sqrt(hits['x']**2 + hits['y']**2), hits['z']))

    # Añadimos de particles el momento p a los hits correspondientes
    hits = hits.merge(particles[['particle_id', 'pt']], on='particle_id', how='left')
    # Mínimo y máximo de p
    print("Mínimo p: {:.2f} GeV/c".format(hits.pt.min()))
    print("Máximo p: {:.2f} GeV/c".format(hits.pt.max()))

    print("Número de partículas totales: {}".format(particles_all.particle_id.nunique()))
    print("Número de partículas únicas con pt<0.5 GeV/C: {}".format(
        hits[hits.pt < 0.5].particle_id.nunique()))

    print(hits.head())
    return hits, particles

def write_metrics_to_log(out_dir , event, accuracy_score, precision_score, y_true,
                        y_pred, recall_score, f1_score, confusion_matrix, classification_report):
    with open(f'{out_dir}/metrics_{event}.txt', 'w') as metrics_file:
        metrics_file.write(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}\n")
        metrics_file.write(f"Precision: {precision_score(y_true, y_pred):.4f}\n")
        metrics_file.write(f"Recall:    {recall_score(y_true, y_pred):.4f}\n")
        metrics_file.write(f"F1-score:  {f1_score(y_true, y_pred):.4f}\n")
        metrics_file.write("\nConfusion Matrix:\n")
        metrics_file.write(f"{confusion_matrix(y_true, y_pred)}\n")
        metrics_file.write("\nClassification Report:\n")
        metrics_file.write(f"{classification_report(y_true, y_pred, digits=4)}\n")

def predict_all(model, loader, device, return_probs=False):
    model.eval()
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits).cpu()
            y_true_all.append(y_batch.float().cpu())

            if return_probs:
                y_pred_all.append(probs)
            else:
                preds = (probs > 0.5).float()
                y_pred_all.append(preds)

    y_true = torch.cat(y_true_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()
    return y_true, y_pred


def training_triplet_model(X, y, model=None, device=None, epochs=500, batch_size=32, lr=1e-3):

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    train_loader, val_loader = create_dataloaders(X, y, batch_size=batch_size, num_workers=4, pin_memory=True)

    print(f"Shape de X: {X.shape}\t Shape de y: {y.shape}\n")

    num_pos = (y == 1).sum().item() if isinstance(y, torch.Tensor) else np.sum(y == 1)
    num_neg = (y == 0).sum().item() if isinstance(y, torch.Tensor) else np.sum(y == 0)
    print(f"Positivos: {num_pos}\t Total negatives: {num_neg}")

    # Cargamos el modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = TripletNet().to(device)
    else:
        model = model.to(device)

    # Ponderación de clases
    PONDERACION = True
    if PONDERACION:
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Directorio de salida y log
    out_dir = './output'
    os.makedirs(out_dir + '/checkpoint', exist_ok=True)

    log = open(out_dir + '/log.train.txt', 'w')
    log.write(f'** start training at {str(datetime.now())} **\n')

    # Cargamos los datos
    train_loader, val_loader = create_dataloaders(X, y, batch_size=batch_size)
    print("Verificando que train_loader funcione correctamente...")
    for batch in train_loader:
        print("Primer batch obtenido correctamente.")
        break

    def evaluate(model, loader):
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y).sum().item()
                total += y.size(0)
        return val_loss / total, val_correct / total  


    # Training loop 
    train_loss, valid_loss = 0, 0

    train_losses, valid_losses, train_accs, valid_accs = [], [], [], []

    model.train()
    start = time.time()

    # Early Stopping
    best_f1 = 0
    epochs_no_improve = 0
    patience = 8
    best_model_state = None


    log.write(' iter   |  valid_loss  valid_acc |  train_loss  train_acc | time\n')
    log.write('---------------------------------------------------------------\n')
    from tqdm import tqdm
    from sklearn.metrics import confusion_matrix

    for epoch in range(epochs):
        sum_loss = 0.0
        correct = 0
        total = 0
        model.train()

        # tqdm para ver cuántos batches llevas de la época
        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(logits) > 0.5).float()

            correct += (preds == y).sum().item()
            total += y.size(0)
            sum_loss += loss.item() * x.size(0)

        # Al terminar la época:
        from sklearn.metrics import f1_score

        y_true_epoch, y_pred_epoch = predict_all(model, val_loader, device)
        f1 = f1_score(y_true_epoch, y_pred_epoch)

        # Calcular matriz de confusión para TP, TN, FP, FN
        tn, fp, fn, tp = confusion_matrix(y_true_epoch, y_pred_epoch).ravel()

        train_loss = sum_loss / total
        train_acc  = correct / total
        v_loss, v_acc = evaluate(model, val_loader)
        print(f"- End Epoch {epoch+1}: "
            f"Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f} | "
            f"Val Loss {v_loss:.4f}, Val Acc {v_acc:.4f} | "
            f"F1 score {f1:.4f}")

        print(f"\tTP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f'\tTime elapsed: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start))}')

        train_losses.append(train_loss)
        valid_losses.append(v_loss)
        train_accs.append(train_acc)
        valid_accs.append(v_acc)

        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping en epoch {epoch+1}: no mejora de F1 en {patience} epochs consecutivos.")
                break


    # print(f'\nEvaluation metrics...\n{evaluate_metrics(model, val_loader)}')

    # Restaurar mejor modelo si se usó early stopping
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    import io
    import contextlib

    # Guardado final
    log.write('\n** Finished Training **\n')


    y_true, y_pred = predict_all(model, val_loader, device)

    print("\n=== Métricas finales en conjunto de validación ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred):.4f}")

    print("\nConfusion Matrix (Validación):")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report (Validación):")
    print(classification_report(y_true, y_pred, digits=4))

    # Guardar las métricas finales en el log
    log.write('\n=== Métricas finales en conjunto de validación ===\n')
    log.write(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}\n")
    log.write(f"Precision: {precision_score(y_true, y_pred):.4f}\n")
    log.write(f"Recall:    {recall_score(y_true, y_pred):.4f}\n")
    log.write(f"F1-score:  {f1_score(y_true, y_pred):.4f}\n")
    log.write("\nConfusion Matrix (Validación):\n")
    log.write(f"{confusion_matrix(y_true, y_pred)}\n")
    log.write("\nClassification Report (Validación):\n")
    log.write(f"{classification_report(y_true, y_pred, digits=4)}\n")

    write_metrics_to_log(out_dir ,'event_combined_precise', accuracy_score, precision_score, y_true,
                        y_pred, recall_score, f1_score, confusion_matrix, classification_report)

    from sklearn.metrics import roc_curve, roc_auc_score

    # Obtener probabilidades
    y_true, y_probs = predict_all(model, val_loader, device, return_probs=True)

    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)

    # Umbral óptimo según Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[optimal_idx]

    # Métricas con mejor umbral
    y_pred_opt = (y_probs > best_thresh).astype(int)

    print(f"\n=== ROC y análisis de umbral ===")
    print(f"AUC: {roc_auc:.4f}")
    print(f"Umbral óptimo: {best_thresh:.4f}")
    print(f"Accuracy (ópt):  {accuracy_score(y_true, y_pred_opt):.4f}")
    print(f"Precision (ópt): {precision_score(y_true, y_pred_opt):.4f}")
    print(f"Recall (ópt):    {recall_score(y_true, y_pred_opt):.4f}")
    print(f"F1-score (ópt):  {f1_score(y_true, y_pred_opt):.4f}")
    print("\nConfusion Matrix (óptimo):")
    print(confusion_matrix(y_true, y_pred_opt))

    # Plot ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], c='red', label=f'Mejor umbral = {best_thresh:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Cierre del log
    log.close()
    GRAPH_PRECISION = True  
    if GRAPH_PRECISION:
        # Graficar las pérdidas y precisiones
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(valid_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(valid_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
   
    return model

import numpy as np
import pandas as pd
def generate_triplets_from_hits(hits, pt_min=0, pt_max=np.inf, cone_angle_fn=None):
    def connection_cone_filter(pairs, layer):
        if cone_angle_fn is None:
            dic_cone_angles = {
                # 0: np.pi / 6.2,   # 5°
                # 1: np.pi / 4.5,   # 6°
                # 2: np.pi / 4,     # 7.5°
                0: np.deg2rad(5),
                1: np.deg2rad(6.5),
                2: np.deg2rad(8)
            }
            cone_angle_aperture = dic_cone_angles.get(layer, np.pi / 4)
        else:
            cone_angle_aperture = cone_angle_fn(pairs['z_in'])  # aplica por vector

        x1 = pairs['x_in']
        y1 = pairs['y_in']
        z1 = pairs['z_in']
        x2 = pairs['x_out']
        y2 = pairs['y_out']
        z2 = pairs['z_out']

        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1

        dot = x1 * dx + y1 * dy + z1 * dz
        norm1 = x1**2 + y1**2 + z1**2
        norm2 = dx**2 + dy**2 + dz**2

        cos_theta = dot / (np.sqrt(norm1 * norm2) + 1e-9)
        cos_z = z1 / (np.sqrt(norm1) + 1e-9)
        sin_z = np.sqrt(np.clip(1 - cos_z**2, 0, 1))

        if isinstance(cone_angle_aperture, float):
            eff_aperture = cone_angle_aperture * sin_z
        else:
            eff_aperture = cone_angle_aperture * sin_z  # array-wise

        cos_alpha = np.cos(eff_aperture / 2)

        return pairs[cos_theta >= cos_alpha].reset_index(drop=True)

    # Etapa 1: Capas 0 y 1
    layer0 = hits[hits['layer'] == 0].reset_index(drop=True)
    layer1 = hits[hits['layer'] == 1].reset_index(drop=True)

    pairs01 = pd.merge(layer0, layer1, on='n_event', suffixes=('_in', '_out'))
    pairs01 = connection_cone_filter(pairs01, layer=0)
    pairs01['label'] = (pairs01['particle_id_in'] == pairs01['particle_id_out']) & (pairs01['particle_id_in'] != 0)

    # Etapa 2: unir con capa 2
    layer2 = hits[hits['layer'] == 2].reset_index(drop=True)

    triplets = pd.merge(pairs01, layer2, on='n_event', suffixes=('', '_3'))
    triplets = triplets.rename(columns={
        'x': 'x3', 'y': 'y3', 'z': 'z3',
        'layer': 'layer3', 'particle_id': 'pid3', 'pt': 'pt3',
        'hit_id': 'hit_id3'
    })

    # Filtro angular entre punto 2 y 3
    x2, y2, z2 = triplets['x_out'], triplets['y_out'], triplets['z_out']
    x3, y3, z3 = triplets['x3'], triplets['y3'], triplets['z3']
    dx, dy, dz = x3 - x2, y3 - y2, z3 - z2

    dot = x2 * dx + y2 * dy + z2 * dz
    norm1 = x2**2 + y2**2 + z2**2
    norm2 = dx**2 + dy**2 + dz**2

    cos_theta = dot / (np.sqrt(norm1 * norm2) + 1e-9)
    cos_z = z2 / (np.sqrt(norm1) + 1e-9)
    sin_z = np.sqrt(np.clip(1 - cos_z**2, 0, 1))

    if cone_angle_fn is None:
        cone_angle_aperture = np.pi / 4
    else:
        cone_angle_aperture = cone_angle_fn(z2)

    eff_aperture = cone_angle_aperture * sin_z
    cos_alpha = np.cos(eff_aperture / 2)

    triplets = triplets[cos_theta >= cos_alpha].reset_index(drop=True)

    # Etiquetas
    triplets['label'] = (
        (triplets['particle_id_in'] == triplets['particle_id_out']) &
        (triplets['particle_id_out'] == triplets['pid3']) &
        (triplets['particle_id_in'] != 0)
    ).astype(int)

    # Filtro pt
    pt_mask = (
        (triplets['pt_in'] >= pt_min) & (triplets['pt_in'] <= pt_max) &
        (triplets['pt_out'] >= pt_min) & (triplets['pt_out'] <= pt_max) &
        (triplets['pt3'] >= pt_min) & (triplets['pt3'] <= pt_max)
    )
    triplets = triplets[pt_mask].reset_index(drop=True)

    # Arrays finales
    coords = triplets[['x_in', 'y_in', 'z_in', 'x_out', 'y_out', 'z_out', 'x3', 'y3', 'z3']].values
    X = coords.reshape(-1, 3, 3)
    y = triplets['label'].values
    pt_values = triplets['pt_in'].values  # podrías usar promedio si prefieres

    return X, y, pt_values


