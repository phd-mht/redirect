from collections import defaultdict


def cw_confusion_matrix(flows_statistics, anomalous_communities, largest_comm_size, anomalies_input, flows_hash_table):
    comms_hash = defaultdict(list)
    for comm_id, comm in anomalous_communities.items():
        for n_id in comm:
            comms_hash[comm_id] += flows_hash_table[n_id]
    comms_hash = {k: set(v) for k, v in comms_hash.items()}
    comms_hash_rev = defaultdict(list)
    for comm_id, f_ids in comms_hash.items():
        for f_id in f_ids:
            comms_hash_rev[f_id].append(comm_id)
    comms_hash_rev = {k: set(v) for k, v in comms_hash_rev.items()}
    
    matches = []
    not_found = []
    for index, stats in flows_statistics.iterrows():
        key = stats["id"]
        turnover_weight = stats["turnover_weight"]
        flow_nodes = set(turnover_weight.keys())
        candidates = []
        for to_match_with in comms_hash_rev.get(key, []):
            matched = flow_nodes.intersection(anomalous_communities[to_match_with])
            matched_score = sum([turnover_weight[x] for x in matched])
            non_matched = set(comm) - matched
            candidates.append((matched_score, matched, non_matched, turnover_weight, comm_id))
        to_score = float(stats["turnover_score"])
        if candidates:
            best = sorted(candidates, reverse=True, key=lambda x: (x[0], -len(x[2])))[0]
            true_positives = best[0]
            false_negatives = len(flow_nodes) - true_positives
            false_positives = len(best[2])
            true_negatives = largest_comm_size - false_positives - len(flow_nodes)
            total = int(round(sum((true_positives, false_negatives, false_positives, true_negatives))))
            assert total == largest_comm_size
            true_positives /= largest_comm_size
            false_positives /= largest_comm_size
            true_negatives /= largest_comm_size
            false_negatives /= largest_comm_size
            matches.append(
                (
                    true_positives * to_score, 
                    false_positives * to_score, 
                    true_negatives * to_score, 
                    false_negatives * to_score, 
                    best[4]
                )
            )
        else:
            not_found.append((0, 0, 0, to_score, key))
    non_matches = set(anomalous_communities.keys()).difference([x[4] for x in matches])
    non_matches = [(x, len(anomalous_communities[x])) for x in non_matches]
    non_matches_pd = anomalies_input.loc[
        anomalies_input.index.isin([x[0] for x in non_matches]), :
    ]

    non_matches_cm = []
    for index, row in non_matches_pd.iterrows():
        to_score = 1
        non_matched = anomalous_communities[index]
        false_positives = len(non_matched)
        true_negatives = largest_comm_size - false_positives
        false_positives /= largest_comm_size
        true_negatives /= largest_comm_size
        non_matches_cm.append((0, false_positives * to_score, true_negatives * to_score, 0, index))

    tp_all, fp_all, tn_all, fn_all = 0, 0, 0, 0
    for tp, fp, tn, fn, _ in matches + non_matches_cm + not_found:
        tp_all += tp
        fp_all += fp
        tn_all += tn
        fn_all += fn

    return tp_all, fp_all, tn_all, fn_all


def cw_recall(tp_all, fn_all):
    return round(tp_all / (tp_all + fn_all), 4)


def cw_f1(tp_all, fp_all, fn_all):
    return round((2 * tp_all) / ((2 * tp_all) + fp_all + fn_all), 4)
