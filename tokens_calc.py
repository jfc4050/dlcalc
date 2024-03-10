from datetime import date

start = date(2024, 4, 15)
end = date(2024, 5, 31)

seqlen = 2048
model_sz = 470e9
n_nodes = 3024
tflops_per_node = 3040
required_tokens_per_day = 47e9
goodput = 0.9

cluster_flops = tflops_per_node * n_nodes * 1e12
flops_per_token = 6 * model_sz
model_flops = seqlen * flops_per_token

required_flops_per_day = required_tokens_per_day * flops_per_token
required_flops_per_sec = required_flops_per_day / (24 * 60 * 60)

required_mfu = required_flops_per_sec / (cluster_flops * goodput)

print(required_mfu)
tokens_trained = (required_tokens_per_day * (end - start).days) * 1e-12
print(tokens_trained)
