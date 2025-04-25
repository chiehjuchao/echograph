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


def entity_with_incoming_relations_reward(hypothesis_annotation_list, reference_annotation_list):
    def build_entity_with_relations(annotation_list):
        entities = annotation_list["entities"]
        incoming_relations = {eid: [] for eid in entities}

        # Track incoming relations
        for eid, entity in entities.items():
            for relation in entity.get("relations", []):
                target_id = relation[1]
                incoming_relations[target_id].append(entity["tokens"].lower())

        entity_sets = set()
        for eid, entity in entities.items():
            has_outgoing_relations = bool(entity.get("relations"))
            if not has_outgoing_relations:
                tokens = [entity["tokens"].lower()] + sorted(incoming_relations[eid])
                entity_representation = "|".join(tokens)
                entity_sets.add((entity_representation, entity["label"]))

        return entity_sets

    hypothesis_set = build_entity_with_relations(hypothesis_annotation_list)
    reference_set = build_entity_with_relations(reference_annotation_list)

    intersection = hypothesis_set & reference_set

    precision = len(intersection) / len(hypothesis_set) if hypothesis_set else 0.0
    recall = len(intersection) / len(reference_set) if reference_set else 0.0

    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) else 0.0
    )

    return f1_score



def weighted_entity_incoming_relations_reward(
    hypothesis_annotation_list, reference_annotation_list, meas_weight=5.0
):
    def build_weighted_entities(annotation_list):
        entities = annotation_list["entities"]
        incoming_relations = {eid: [] for eid in entities}

        # Track incoming relations
        for eid, entity in entities.items():
            for relation in entity.get("relations", []):
                target_id = relation[1]
                incoming_relations[target_id].append(entities[eid])

        entity_dict = dict()
        for eid, entity in entities.items():
            has_outgoing_relations = bool(entity.get("relations"))
            if not has_outgoing_relations:
                # Start with the main entity
                tokens = [entity["tokens"].lower()]
                weight = meas_weight if entity["label"] == "MEAS" else 1

                # Add incoming related entities
                for incoming_entity in incoming_relations[eid]:
                    tokens.append(incoming_entity["tokens"].lower())
                    if incoming_entity["label"] == "MEAS":
                        weight += (meas_weight - 1)  # additional weight if incoming entity is MEAS

                entity_representation = "|".join(sorted(tokens))
                entity_dict[(entity_representation, entity["label"])] = weight

        return entity_dict

    hyp_entities = build_weighted_entities(hypothesis_annotation_list)
    ref_entities = build_weighted_entities(reference_annotation_list)

    hyp_set = set(hyp_entities.keys())
    ref_set = set(ref_entities.keys())
    intersection = hyp_set & ref_set

    # Compute weighted true positives, total hypothesis, and total reference
    tp_weighted = sum(hyp_entities[e] for e in intersection)
    total_hyp_weighted = sum(hyp_entities.values())
    total_ref_weighted = sum(ref_entities.values())

    precision = tp_weighted / total_hyp_weighted if total_hyp_weighted else 0.0
    recall = tp_weighted / total_ref_weighted if total_ref_weighted else 0.0

    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) else 0.0
    )

    return f1_score




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
    hierarchical = weighted_entity_incoming_relations_reward(hypothesis_annotation_list, reference_annotation_list)

    all = (weighted_gauge, harmonic, hierarchical)

    return eval(reward_level)

