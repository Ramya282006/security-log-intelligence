
LANL TRANSFORMER DATASET

Sequence count: 115
Maximum sequence length: 80

Feature order:
0. user_id
1. computer_id
2. process_id
3. event_type_id
4. time_delta
5. event_duration
6. event_count
7. sequence_position

Next-event prediction:
Train: 1331 examples
Validation: 226 examples
Test: 131 examples

Each NPZ file contains:
categorical_inputs
numeric_inputs
attention_masks
targets

Categorical target order:
user_id, computer_id, process_id, event_type_id

Use lanl_vocabularies.json for categorical mappings.

Preserve the existing group-based data splits.
Do not use redteam_time, time_to_redteam, incident_id,
overlap_group_id or data_split as model input.

LANL candidates are red-team-associated evidence,
not verified binary attack labels.
This dataset supports exploratory next-event prediction,
not supervised attack-vs-normal classification.
