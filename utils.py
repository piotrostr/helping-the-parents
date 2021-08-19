
def get_in_ms(x):
    try: 
        return (float(x) - 9.257890218838680000e+14) * 1e-6
    except:
        return x


def make_weird(x):
    try:
        return (float(x) / 1e-6) + 9.257890218838680000e+14
    except:
        return x


def list_keys(inp):
    for k in inp.keys():
        try:
            print(f'{k:<30} {inp[k].shape}')
        except:
            print(inp[k])

