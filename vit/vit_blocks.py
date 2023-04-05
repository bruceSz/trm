import torch
import torch.nn as nn
import torch.nn.functional as F


def img2emb_naive( img, patch_size, weight):
    """
        img: shape of bs * channels * h * w
        patch_size: size of slicing window
        weight: weight of linear transformation.
    """

    patch = F.unfold(img, kernel_size = patch_size, stride=patch_size).transpose(-1,-2)
    print("patch size: ", patch.shape)
    patch_emb = patch @ weight
    return patch_emb



def img2emb_conv(img, kernel, stride):
    conv_out = F.conv2d(img, kernel, stride=stride)
    # bs * oc * h * w
    bs, oc, h, w = conv_out.shape
    print("conv out h,w: ", h, w)
    emb_ = conv_out.reshape([bs, oc, h * w]).transpose(-1,-2)
    return emb_
    


def test_img2emb():
    bs , ic, h, w = 1,3, 8, 8
    patch_size = 4
    model_dim = 8
    img = torch.randn(bs, ic, h, w)
    #example label
    label = torch.randint(10,(bs,))
    #weight = torch.randn(None, model_dim)
    patch_depth =  patch_size * patch_size  * ic
    weight = torch.randn(patch_depth, model_dim)
    print("shape of weight:", weight.shape)
    emb_naive = img2emb_naive(img, patch_size, weight)
    print("emb naive shape: ", emb_naive.shape)


    kernel = weight.transpose(0,1).reshape((-1, ic, patch_size, patch_size))

    emb_conv = img2emb_conv(img, kernel, stride =patch_size)
    print("emb conv shape: ", emb_conv.shape)
    print(emb_conv)

    #prepend CLS token embedding
    cls_tokenembedding = torch.randn(bs, 1, model_dim).requires_grad_(True)
    token_embedding = torch.cat([cls_tokenembedding, emb_conv],dim=1)
    print("token embedding shape: ", token_embedding.shape)

    # add position embedding
    max_num_token = 16

    pos_embedding = torch.randn(max_num_token, model_dim).requires_grad_(True) 
    seq_len = token_embedding.shape[1]
    #shape align with token_embeddin
    pos_embedding =  torch.tile(pos_embedding[:seq_len], [token_embedding.shape[0], 1, 1])
    token_embedding += pos_embedding
    
    # step 4 pass embedding to transformer

    encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=8)
    t_encoder = nn.TransformerEncoder(encoder_layer, num_layers=16)
    encoder_out = t_encoder(token_embedding)


    # step 5 do classification prediction

    cls_token_o = encoder_out[:,0,:]
    linear_layer = nn.Linear(model_dim, 10)
    logits = linear_layer(cls_token_o)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits,label)
    print(loss)

    

    
if __name__ == "__main__":
    test_img2emb()