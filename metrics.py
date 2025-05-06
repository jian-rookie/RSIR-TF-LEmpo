import numpy as np


def metrics_calc(sim_matrix, query2gallery=None, at=[1, 5, 10]):
    """
    sim: temsor (query_num, gallery_num)
    calulate metrics Average Precision, Recall@k, and Precision@k for a given similarity tensor
    """
    metrics = {f"R@{k}": [] for k in at}
    metrics.update({f"P@{k}": [] for k in at})
    metrics["AP"] = []

    sim_matrix = sim_matrix.detach().cpu().numpy()
    for idx, sim in enumerate(sim_matrix):
        retrieved_indices = np.argsort(sim)[::-1] # 降序排列的index
        gt = query2gallery[idx]
        gt_indices = np.where(gt == 1)[0]
        is_relevant = np.isin(retrieved_indices, gt_indices).astype(int)
        
        precisions = []
        relevant_count = 0
        cnt = 0
        for k, rel in enumerate(is_relevant, start=1):
            cnt += 1
            if rel:
                relevant_count += 1
                precisions.append(relevant_count / k)
        ap = sum(precisions) / len(precisions) if precisions else 0
        metrics["AP"].append(ap * 100)

        total_relevant = sum(is_relevant)
        for k in at:
            relevant_at_k = sum(is_relevant[:k])
            precision_at_k = relevant_at_k / k if k else 0
            recall_at_k = relevant_at_k / total_relevant if total_relevant else 0

            metrics[f"P@{k}"].append(precision_at_k * 100)
            metrics[f"R@{k}"].append(recall_at_k * 100)

    for k, v in metrics.items():
        metrics[k] = np.mean(v)
    metrics['mR'] = np.mean([metrics[f"R@{k}"] for k in at])

    for k, v in metrics.items():
        metrics[k] = round(v, 2)
    return metrics