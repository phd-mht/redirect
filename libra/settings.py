G_FLOW_PREFIX = "graph_flow_feat_"
G_COMM_THRESHOLD_PREFIX = "graph_comm_threshold_feat_"
G_COMM_LIMIT_PREFIX = "graph_comm_limit_feat_"
G_1HOP_PREFIX = "graph_1_hop_feat_"
G_GLOB_PREFIX = "graph_global_"

ALL_PREFIXES = [G_FLOW_PREFIX, G_COMM_THRESHOLD_PREFIX, G_COMM_LIMIT_PREFIX, G_1HOP_PREFIX, G_GLOB_PREFIX]
assert len(ALL_PREFIXES) == len(set(ALL_PREFIXES))
