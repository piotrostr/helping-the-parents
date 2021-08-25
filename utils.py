import torch



def get_sample_input():
    from EVE.src.datasources.eve_sequences import EVESequencesBase
    dataset = EVESequencesBase(
        'sample/eve_dataset',
        participants_to_use=['train01']
    )
    dataloader = torch.utils.data.DataLoader(dataset)
    inp = next(iter(dataloader))
    return inp

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


def true_tensor(n):
    return torch.Tensor([True for i in range(n)]).type(torch.bool)


def save_heatmap_plot(out):
    assert 'final_heatmap' in out.keys()
    heatmap = out['final_heatmap']
    heatmap = heatmap.squeeze(0).transpose(1, 2).transpose(0, 2) * 255
    heatmap = heatmap.numpy().astype(int)
    path = 'heatmap.jpg'
    cv2.imwrite(path, heatmap)
    print(f'Saved {path}')

