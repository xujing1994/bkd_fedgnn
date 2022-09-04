num_epochs = 5
num_workers = 10

idx_max_length = 50000
grad_shift = 2 ** 20

f = 1

topk = 40

gradient_frac = 2 ** 10
gradient_rand = 2 ** 8

server1_address = "127.0.0.1"
port1 = 50001


server2_address = "127.0.0.1"
port2 = 50002


mpc_idx_port = 50003
mpc_grad_port = 50004

grpc_options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]