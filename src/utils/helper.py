def parse_mode_type(dmode: str):
    type_dict = {'target_type_str': dmode[:3],
                'cond_str': dmode.split('-')[2],
                'loss_type_str': dmode[-2:],
                }
    return type_dict

def zip_diffusion_input(x, prior_z, post_z, dict):
    """_summary_

    Args:
        x (NA, HT, 4): _description_
        prior_z (NA, FT): _description_
        post_z (NA, FT): _description_
        dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert(dict['target_type_str'] in ['z2z', 'x2x'])
    if dict['target_type_str'] == 'z2z':
        target_signal = post_z.unsqueeze(2) 
    elif dict['target_type_str'] == 'x2x':
        target_signal = x

    assert(dict['cond_str'] in ['none', 'z'])   
    if dict['cond_str'] == 'none':
        cond = target_signal
    elif dict['cond_str'] == 'z':
        cond = prior_z    

    return target_signal, cond  