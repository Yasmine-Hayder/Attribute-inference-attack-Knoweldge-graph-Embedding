import numpy as np
import torch
import random
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

KG_PATH = "knowledge_graph_triples.txt"
MODEL_PATH = "results/trained_model_rotatE_CPU/trained_model.pkl"

FRACTION = 0.1
N_RUNS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_triples(path):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            h, r, t = line.strip().split("\t")
            triples.append((h, r, t))
    return np.array(triples, dtype=str)


def load_recommender_model(model_path, device):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    return model


def score_all_movies(model, tf, user, device):
    ent2id = tf.entity_to_id
    if user not in ent2id:
        return [], torch.tensor([])

    user_id = ent2id[user]
    r_id = tf.relation_to_id["rated"]

    movie_entities = [e for e in ent2id if e.startswith("movie_")]
    movie_ids = torch.tensor([ent2id[m] for m in movie_entities], device=device)

    h_ids = torch.full((len(movie_ids),), user_id, device=device)
    r_ids = torch.full((len(movie_ids),), r_id, device=device)

    triples = torch.stack([h_ids, r_ids, movie_ids], dim=1)

    with torch.no_grad():
        scores = model.score_hrt(triples).view(-1)

    s_min = scores.min()
    s_max = scores.max()
    scores = (scores - s_min) / (s_max - s_min + 1e-8)

    return movie_entities, scores


def select_attack_users(triples, fraction):
    females, males = [], []

    for h, r, t in triples:
        if r == "has_gender":
            if t == "f":
                females.append(h)
            elif t == "m":
                males.append(h)

    n = int(fraction * min(len(females), len(males)))
    return random.sample(females, n) + random.sample(males, n)


def replace_history_with_2rec_3random(triples, tf, model, attack_users, device):
    new_triples = []
    utilities = []

    all_movies_global = sorted({t for h, r, t in triples if r == "rated" and t.startswith("movie_")})

    for h, r, t in triples:
        if h in attack_users and r == "rated":
            continue
        new_triples.append((h, r, t))

    for u in attack_users:
        movie_entities, scores = score_all_movies(model, tf, u, device)

        if len(movie_entities) == 0:
            continue

        movie_to_score = {movie_entities[i]: scores[i].item() for i in range(len(movie_entities))}

        top10_idx = torch.topk(scores, min(10, len(scores))).indices
        top10_movies = [movie_entities[i.item()] for i in top10_idx]

        shuffled = top10_movies.copy()
        rec_movies = shuffled[:7]

        remaining = list(set(all_movies_global) - set(rec_movies))
        rand_movies = random.sample(remaining, 3)

        final_movies = rec_movies + rand_movies

        sanitized_sum = sum(movie_to_score[m] for m in final_movies if m in movie_to_score)

        top5_idx = torch.topk(scores, min(10, len(scores))).indices
        top5_sum = scores[top5_idx].sum().item()

        if top5_sum > 0:
            utilities.append(sanitized_sum / top5_sum)

        for m in final_movies:
            new_triples.append((u, "rated", m))

    Q_mean = float(np.mean(utilities)) if utilities else 0.0
    return np.array(new_triples, dtype=str), Q_mean


def split_attack(triples, attack_users):
    train, test = [], []

    for h, r, t in triples:
        if h in attack_users and r == "has_gender":
            test.append((h, r, t))
        else:
            train.append((h, r, t))

    return np.array(train, dtype=str), np.array(test, dtype=str)


def train_attack_model(train_tf, test_tf, seed):
    result = pipeline(
        model="RotatE",
        training=train_tf,
        testing=test_tf,
        model_kwargs=dict(embedding_dim=70),
        training_kwargs=dict(num_epochs=100, batch_size=64),
        loss="MarginRankingLoss",
        loss_kwargs=dict(margin=1.0),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=5e-4),
        device=DEVICE,
        random_seed=seed,
    )
    return result.model


def evaluate_attack(model, train_tf, test_tf):
    model = model.to("cpu")
    model.eval()

    ent2id = train_tf.entity_to_id
    id2ent = train_tf.entity_id_to_label
    gender_ids = torch.tensor([ent2id["f"], ent2id["m"]])

    correct = 0

    for h_id, r_id, t_id in test_tf.mapped_triples:
        h_ids = torch.tensor([h_id, h_id])
        r_ids = torch.tensor([r_id, r_id])
        triples = torch.stack([h_ids, r_ids, gender_ids], dim=1)

        with torch.no_grad():
            scores = model.score_hrt(triples)

        pred = ["f", "m"][torch.argmax(scores).item()]
        true = id2ent[int(t_id)]
        correct += int(pred == true)

    return correct / len(test_tf.mapped_triples)


accuracies = []
utilities = []

for run in range(N_RUNS):
    seed = random.randint(0, 1_000_000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n===== RUN {run+1}/{N_RUNS} (seed={seed}) =====")

    triples = load_triples(KG_PATH)
    attack_users = select_attack_users(triples, FRACTION)

    base_tf = TriplesFactory.from_labeled_triples(triples)
    reco_model = load_recommender_model(MODEL_PATH, DEVICE)

    triples_modified, Q_mean = replace_history_with_2rec_3random(
        triples,
        base_tf,
        reco_model,
        attack_users,
        device=DEVICE,
    )

    train, test = split_attack(triples_modified, attack_users)

    train_tf = TriplesFactory.from_labeled_triples(train)
    test_tf = TriplesFactory.from_labeled_triples(
        test,
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
    )

    model = train_attack_model(train_tf, test_tf, seed)
    acc = evaluate_attack(model, train_tf, test_tf)

    accuracies.append(acc)
    utilities.append(Q_mean)

    print(f"Run accuracy: {acc:.3f}")
    print(f"Run utility Q: {Q_mean:.4f}")

print("\n==============================")
print("All accuracies:", accuracies)
print(f"Mean accuracy: {np.mean(accuracies):.4f}")
print(f"Std deviation: {np.std(accuracies):.4f}")
print("All utilities:", utilities)
print(f"Mean utility Q: {np.mean(utilities):.4f}")
print(f"Std utility Q: {np.std(utilities):.4f}")
print("==============================")
