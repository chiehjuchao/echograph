def exact_entity_token_if_all_match_reward(
        hypothesis_annotation_list, reference_annotation_list
):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            if not entity["relations"]:
                candidate.append((entity["tokens"], entity["label"]))
            if entity["relations"]:
                candidate.extend([(entity["tokens"].lower(),
                                   entity["label"],
                                   r[0],
                                   annotation_list["entities"][r[1]]["tokens"].lower())
                                  for r in entity["relations"]]
                                 )

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates
    precision = (
        sum(
            [
                1
                for x in hypothesis_relation_token_list
                if (x in reference_relation_token_list)
            ]
        )
        / len(hypothesis_relation_token_list)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum(
            [
                1
                for x in reference_relation_token_list
                if (x in hypothesis_relation_token_list)
            ]
        )
        / len(reference_relation_token_list)
        if len(reference_relation_token_list) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score


def exact_entity_token_if_rel_exists_reward(
        hypothesis_annotation_list, reference_annotation_list
):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            if not entity["relations"]:
                candidate.append((entity["tokens"], entity["label"]))
            if entity["relations"]:
                candidate.append((entity["tokens"], entity["label"], True))

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates

    precision = (
        sum(
            [
                1
                for x in hypothesis_relation_token_list
                if (x in reference_relation_token_list)
            ]
        )
        / len(hypothesis_relation_token_list)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum(
            [
                1
                for x in reference_relation_token_list
                if (x in hypothesis_relation_token_list)
            ]
        )
        / len(reference_relation_token_list)
        if len(reference_relation_token_list) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score


def exact_entity_token_match_reward(
        hypothesis_annotation_list, reference_annotation_list
):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            candidate.append((entity["tokens"], entity["label"]))

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates

    precision = (
        sum(
            [
                1
                for x in hypothesis_relation_token_list
                if (x in reference_relation_token_list)
            ]
        )
        / len(hypothesis_relation_token_list)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum(
            [
                1
                for x in reference_relation_token_list
                if (x in hypothesis_relation_token_list)
            ]
        )
        / len(reference_relation_token_list)
        if len(reference_relation_token_list) > 0
        else 0.0
    )

    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score


def weighted_gauge_meas_f1_reward(
    hypothesis_annotation_list, reference_annotation_list, alpha=10.0
):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            is_meas_gauge = False
            for relation in entity.get("relations", []):
                related_entity = annotation_list["entities"][relation[1]]
                if entity["label"] == "MEAS" and relation[0] == "Gauge":
                    candidate.append((
                        entity["tokens"].lower(),
                        entity["label"],
                        relation[0],
                        related_entity["tokens"].lower(),
                        "MEAS_GAUGE"
                    ))
                    is_meas_gauge = True

            if not entity["relations"] or not is_meas_gauge:
                if not entity["relations"]:
                    candidate.append((entity["tokens"].lower(), entity["label"]))
                else:
                    candidate.extend([
                        (entity["tokens"].lower(), entity["label"], r[0],
                         annotation_list["entities"][r[1]]["tokens"].lower())
                        for r in entity["relations"]
                    ])

        candidates.append(set(candidate))

    hypothesis_set, reference_set = candidates
    intersection = hypothesis_set & reference_set

    def weighted_count(token_set, weight_meas_gauge):
        return sum(weight_meas_gauge if len(token) > 4 and token[-1] == "MEAS_GAUGE" else 1 for token in token_set)

    tp = weighted_count(intersection, alpha)
    precision = tp / weighted_count(hypothesis_set, alpha) if hypothesis_set else 0.0
    recall = (
        weighted_count(intersection, alpha) / weighted_count(reference_set, alpha)
        if reference_set else 0.0
    )

    return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0


def meas_gauge_reward(hypothesis_annotation_list, reference_annotation_list):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            if entity["label"] == "MEAS":
                for relation in entity["relations"]:
                    if relation[0] == "Gauge":
                        related_entity = annotation_list["entities"][relation[1]]
                        candidate.append((
                            entity["tokens"].lower(),
                            entity["label"],
                            relation[0],
                            related_entity["tokens"].lower()
                        ))
        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates
    intersection = hypothesis_relation_token_list & reference_relation_token_list

    precision = (
        len(intersection) / len(hypothesis_relation_token_list)
        if hypothesis_relation_token_list else 0.0
    )
    recall = (
        len(intersection) / len(reference_relation_token_list)
        if reference_relation_token_list else 0.0
    )

    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) else 0.0
    )

    return f1_score
        

def weighted_harmonic_mean_reward(hypothesis_annotation_list, reference_annotation_list, beta=2.0):
    gauge_reward = meas_gauge_reward(hypothesis_annotation_list, reference_annotation_list)
    textual_reward = exact_entity_token_match_reward(hypothesis_annotation_list, reference_annotation_list)

    if gauge_reward == 0 or textual_reward == 0:
        return 0.0

    beta_sq = beta ** 2
    return (1 + beta_sq) * (gauge_reward * textual_reward) / (beta_sq * gauge_reward + textual_reward)


def hierarchical_reward(hypothesis_annotation_list, reference_annotation_list):
    gauge_reward = meas_gauge_reward(hypothesis_annotation_list, reference_annotation_list)
    textual_reward = exact_entity_token_match_reward(hypothesis_annotation_list, reference_annotation_list)

    if gauge_reward < 1.0:
        return gauge_reward * textual_reward
    else:
        return (gauge_reward + textual_reward) / 2.0


def compute_reward(
        hypothesis_annotation_list,
        reference_annotation_list,
        reward_level,
        alpha=10.0,
        beta=2.0
):
    assert reward_level in ["weighted_gauge", "harmonic", "hierarchical", "all"]

    if (
            len(hypothesis_annotation_list["entities"].keys()) == 0
            or len(reference_annotation_list["entities"].keys()) == 0
    ):
        return (0., 0., 0.) if reward_level == "all" else 0.

    weighted_gauge = weighted_gauge_meas_f1_reward(hypothesis_annotation_list, reference_annotation_list, alpha)
    harmonic = weighted_harmonic_mean_reward(hypothesis_annotation_list, reference_annotation_list, beta)
    hierarchical = hierarchical_reward(hypothesis_annotation_list, reference_annotation_list)

    all = (weighted_gauge, harmonic, hierarchical)

    return eval(reward_level)

