def get_activation(mem, name):
    def get_output_hook(module, input, output):
        mem[name] = output
    return get_output_hook

def add_hook(net, mem, mapping_layers):
    for n, m in net.named_modules():
        if n in mapping_layers:
            m.register_forward_hook(get_activation(mem, n))
