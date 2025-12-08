"""
Parameters shared between different modules

Date:
    - Jan. 28, 2023
"""
PAD_IDX = 262  # from params or data module
MAX_LENGTH = 64
SOS_token = 260
EOS_token = 259

RAND_SEED = 42
teacher_forcing_ratio = 0.5


# data-related parameters
num_node_init_feats = 12
num_layer_dec = 1

# output paths
tr_loss_fig_name = "outputs/training_loss.png"
eval_out_attn_fig_name = "outputs/eval_out_attns.png"


# input related
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)
