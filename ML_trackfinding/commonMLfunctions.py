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
class TripletDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Dataset y dataloaders
def create_dataloaders(X, y, batch_size, seed=42):
    dataset = TripletDataset(X, y)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Establece la semilla para reproducibilidad
    generator = torch.Generator().manual_seed(seed)
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Dataset original length:", len(dataset))
    print("Train subset length:", len(train_dataset))
    print("Val subset length:", len(val_dataset))

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

def predict_all(model, loader, device):
    model.eval()
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu()
            y_true_all.append(y_batch.cpu())
            y_pred_all.append(preds)
    y_true = torch.cat(y_true_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()
    return y_true, y_pred

def training_triplet_model(event, model=None):
    # Sacamos los valores de X y y
    data_dir = '/mnt/d/TFG - Dataset/OUTPUT'  # Ruta montada en WSL
    X = np.load(os.path.join(data_dir, f'triplets_data_{event}.npz'))['X']
    y = np.load(os.path.join(data_dir, f'triplets_data_{event}.npz'))['y']


    print(f"Shape de X: {X.shape}\t Shape de y: {y.shape}\n")

    num_pos = (y == 1).sum().item() if isinstance(y, torch.Tensor) else np.sum(y == 1)
    num_neg = (y == 0).sum().item() if isinstance(y, torch.Tensor) else np.sum(y == 0)
    print(f"Positivos: {num_pos}\t Total negatives: {num_neg}")

    # Aseguramos que X y y son tensores de PyTorch
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Parámetros adicionales
    batch_size = 32
    lr = 1e-5
    epochs = 500

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

    iter_accum = 1
    iter_valid = len(train_loader)

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
    j = 0  # steps
    i = 0  # iteraciones acumuladas

    train_losses, valid_losses, train_accs, valid_accs = [], [], [], []

    model.train()
    start = time.time()

    # Early Stopping
    best_f1 = 0
    epochs_no_improve = 0
    patience = 10
    best_model_state = None


    log.write(' iter   |  valid_loss  valid_acc |  train_loss  train_acc | time\n')
    log.write('---------------------------------------------------------------\n')

    for epoch in range(epochs):
        sum_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            if j % iter_accum == 0:
                optimizer.step()

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
            sum_loss += loss.item() * x.size(0)

            if j % iter_valid == 0 and j > 0:
                v_loss, v_acc = evaluate(model, val_loader)
                train_loss = sum_loss / total
                train_acc = correct / total
                t = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))

                log.write(f'{j:6d} |  {v_loss:.4f}     {v_acc:.4f} |  {train_loss:.4f}     {train_acc:.4f} | {t}\n')
                log.flush()

                sum_loss = 0.0
                correct = 0
                total = 0

            j += 1

        # Al final del epoch calculamos métricas finales (entrenamiento y validación)
        train_loss = sum_loss / total if total > 0 else 0
        train_acc = correct / total if total > 0 else 0
        v_loss, v_acc = evaluate(model, val_loader)

        print(f'- Epoch {epoch+1}/{epochs} completed.')
        print(f'\tTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'\tValidation Loss: {v_loss:.4f}, Validation Acc: {v_acc:.4f}')
        print(f'\tTime elapsed: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start))}')
        train_losses.append(train_loss)
        valid_losses.append(v_loss)
        train_accs.append(train_acc)
        valid_accs.append(v_acc)

        from sklearn.metrics import f1_score

        y_true_epoch, y_pred_epoch = predict_all(model, val_loader, device)
        f1 = f1_score(y_true_epoch, y_pred_epoch)

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
    SAVE_PER_EVENT = False
    if SAVE_PER_EVENT:
        torch.save(model.state_dict(), f'{out_dir}/checkpoint/final_model_{event}.pth')
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

    write_metrics_to_log(out_dir , event, accuracy_score, precision_score, y_true,
                        y_pred, recall_score, f1_score, confusion_matrix, classification_report)

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


