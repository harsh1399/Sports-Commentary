
class config :
    ENCODER = "google/vivit-b-16x2-kinetics400"
    # DECODER = "gpt2"
    DECODER = "mistralai/Mistral-7B-v0.1"
    TRAIN_BATCH_SIZE = 2
    VAL_BATCH_SIZE = 2
    VAL_EPOCHS = 1
    LR = 5e-5
    SEED = 42
    MAX_LEN = 128
    SUMMARY_LEN = 20
    WEIGHT_DECAY = 0.01
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    TRAIN_PCT = 0.95
    EPOCHS = 5
    IMG_SIZE = (224,224)
    LABEL_MASK = -100
    TOP_K = 1000
    TOP_P = 0.95
