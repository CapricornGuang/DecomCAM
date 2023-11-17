import numpy as np

def get_singleImage_attnMap(cam, model, layer, inputs):
    '''
    cam: the cam-method class
    '''
    #Call CAM
    assert len(inputs) <= 3, 'Inputs exceeds the limited CAM supported number'
    image_input, text_input = inputs['image_input'], inputs['text_input']
    if len(inputs) == 2: #gradient based method
        cam_output = cam(image_input.half(), text_input.float())
    elif len(inputs) == 3: #score based method
        image_np = inputs['image_np']
        cam_output = cam(image_input.half(), text_input.float(), image_np)
    
    
    attn_map = cam_output['attn_map']
    if isinstance(attn_map, np.ndarray):
        pass
    else:
        attn_map = attn_map.squeeze().detach().cpu().numpy()

    return attn_map