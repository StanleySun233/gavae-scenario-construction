from pathlib import Path


DATA_ROOT = Path("data")
DATASET_ROOT = DATA_ROOT
TRAIN_RUN_ROOT = DATA_ROOT / "runs" / "train"
TEST_RUN_ROOT = DATA_ROOT / "runs" / "test"
GENERATE_RUN_ROOT = DATA_ROOT / "runs" / "generate"
SCENARIO_RUN_ROOT = DATA_ROOT / "runs" / "scenario"

EPOCHS = 2000
BATCH_SIZE = 770
# Training and scenario construction batch sizes are different workloads.
SCENARIO_BATCH_SIZE = 64
SCENARIO_EARLY_SECONDS = 100.0
SCENARIO_AFTER_SECONDS = 100.0
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.1
SAMPLES = 1000
PRECISION = "float32"
CORRIDOR_CALIBRATION_BLEND = 0.8
