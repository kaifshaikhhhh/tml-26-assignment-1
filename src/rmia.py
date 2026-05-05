from utils import compute_model_probabilities

# Function to compute RMIA scores following the algorithm
def get_rmia_scores(model_target, reference_models, target_loader, Z_loader, gamma=1.0, a=0.3, device=None):
    k = len(reference_models)

    # Compute Pr(x|θ) for x in target_loader
    Pr_x_given_theta = compute_model_probabilities(model_target, target_loader, device=device)

    # Compute Pr(z|θ) for z in Z_loader
    Pr_z_given_theta = compute_model_probabilities(model_target, Z_loader, device=device)

    # Compute Pr(x|θ') for θ' ∈ Θ and x ∈ target_loader
    Pr_x_given_ref_models = {}
    for i, ref_model in enumerate(reference_models, start=1):
        print(f"Scoring target data with reference model {i}/{k}")
        probs = compute_model_probabilities(ref_model, target_loader, device=device)
        for id_, prob in probs.items():
            if id_ not in Pr_x_given_ref_models:
                Pr_x_given_ref_models[id_] = []
            Pr_x_given_ref_models[id_].append(prob)

    # Compute Pr(z|θ') for θ' ∈ Θ and z ∈ Z_loader
    Pr_z_given_ref_models = {}
    for i, ref_model in enumerate(reference_models, start=1):
        print(f"Scoring Z data with reference model {i}/{k}")
        probs = compute_model_probabilities(ref_model, Z_loader, device=device)
        for id_, prob in probs.items():
            if id_ not in Pr_z_given_ref_models:
                Pr_z_given_ref_models[id_] = []
            Pr_z_given_ref_models[id_].append(prob)

    # Compute Pr(x) for each x
    Pr_x = {}
    for x_id in Pr_x_given_ref_models:
        Pr_x_OUT = sum(Pr_x_given_ref_models[x_id]) / k
        Pr_x[x_id] = 0.5 * ((1 + a) * Pr_x_OUT + (1 - a))

    # Compute Pr(z) for each z
    Pr_z = {}
    for z_id in Pr_z_given_ref_models:
        Pr_z_OUT = sum(Pr_z_given_ref_models[z_id]) / k
        Pr_z[z_id] = 0.5 * ((1 + a) * Pr_z_OUT + (1 - a))

    # Compute Ratio_x and Ratio_z
    Ratio_x = {}
    for x_id in Pr_x:
        # Add a small epsilon to avoid division by zero
        Ratio_x[x_id] = Pr_x_given_theta[x_id] / (Pr_x[x_id] + 1e-10)

    Ratio_z = {}
    for z_id in Pr_z:
        Ratio_z[z_id] = Pr_z_given_theta[z_id] / (Pr_z[z_id] + 1e-10)

    # Compute scores
    scores = {}
    for x_id in Ratio_x:
        C = 0
        Rx = Ratio_x[x_id]
        for Rz in Ratio_z.values():
            if (Rx / (Rz + 1e-10)) > gamma:
                C += 1
        ScoreMIA = C / len(Ratio_z)
        scores[x_id] = ScoreMIA

    return scores
