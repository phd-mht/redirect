import os


TIMESTAMP_FORMAT = "yyyy/MM/dd HH:mm"
VALID_SIZES = ["Small", "Medium", "Large"]

HIGH_ILLICIT = True
FILE_SIZE = "Large"

assert FILE_SIZE in VALID_SIZES
ILLICIT_TYPE = "HI" if HIGH_ILLICIT else "LI"
MAIN_LOCATION = os.path.join(os.path.curdir, "data")
DATA_FILE = os.path.join(MAIN_LOCATION, f"{ILLICIT_TYPE}-{FILE_SIZE}_Trans.csv")
PATTERNS_FILE = os.path.join(MAIN_LOCATION, f"{ILLICIT_TYPE}-{FILE_SIZE}_Patterns.txt")

OUTPUT_POSTFIX = f"-{ILLICIT_TYPE.lower()}-{FILE_SIZE.lower()}"
STAGED_DATA_LOCATION = os.path.join(MAIN_LOCATION, f"staged-transactions{OUTPUT_POSTFIX}")
STAGED_CASES_DATA_LOCATION = os.path.join(
    MAIN_LOCATION, f"staged-cases-transactions{OUTPUT_POSTFIX}.parquet"
)
STAGED_DATA_CSV_LOCATION = os.path.join(MAIN_LOCATION, f"staged-transactions{OUTPUT_POSTFIX}.csv")
STAGED_PATTERNS_CSV_LOCATION = os.path.join(MAIN_LOCATION, f"staged-patterns{OUTPUT_POSTFIX}.txt")

G_FLOW_PREFIX = "graph_flow_feat_"
G_COMM_THRESHOLD_PREFIX = "graph_comm_threshold_feat_"
G_COMM_LIMIT_PREFIX = "graph_comm_limit_feat_"
G_1HOP_PREFIX = "graph_1_hop_feat_"
G_GLOB_PREFIX = "graph_global_"

ALL_PREFIXES = [G_FLOW_PREFIX, G_COMM_THRESHOLD_PREFIX, G_COMM_LIMIT_PREFIX, G_1HOP_PREFIX, G_GLOB_PREFIX]
assert len(ALL_PREFIXES) == len(set(ALL_PREFIXES))
