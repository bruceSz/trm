import torch
from torch import nn
def lstm():
    bs, T, i_size, h_size  = 2,3,4,5
    #proj_size = 
    input = torch.randn(bs, T, i_size)
    # initial state, not trainable.
    c_0 = torch.randn(bs, h_size)
    h_0 = torch.randn(bs, h_size)

    lstm_layer = nn.LSTM(i_size, h_size, batch_first=True)
    out, (h_n, c_n) = lstm_layer(input,(h_0.unsqueeze(0), c_0.unsqueeze(0))) 

    print(out)
    for k,v in lstm_layer.named_parameters():
        print(k, v.shape)


    
def lstm_forward(input, init_state, w_ih, w_hh, b_ih, b_hh):
    h0, c0 = init_state
    bs, T, i_size = input.shape
    h_size = w_ih.shape[0]//4

    prev_h = h0
    prev_c = c0

    output_size = h_size

    output = torch.zeros(bs, T, output_size)
    # [bs, 4 * h_size, i_size]
    batch_w_ih = w_ih.unsqueeze(0).tile(bs,1,1) # 4 * h_size , i_size
    # [bs, 4*h_size, h_size]
    batch_w_hh  = w_hh.unsqueeze(0).tile(bs, 1,1)# 4 * h_size , h_size

    for t in range(T):
        #[bs, i_size]
        x = input[:,t,:]
        #[bs, 4 * h_size, 1]
        w_times_x = torch.bmm(batch_w_ih, x.unsqueeze(-1))
        w_times_x  = w_times_x.squeeze(-1)

        w_times_h_prev = torch.bmm(batch_w_hh, x.unsqueeze(-1))
        w_times_h_prev = w_times_h_prev.squeeze(-1)

        # input gate
        i_t = torch.sigmoid( w_times_x[:, :h_size] + w_times_h_prev[:, ] 
                + b_ih[:h_size] + b_hh[:h_size])


        # forget gate

        f_t = torch.sigmoid( w_times_x[:, h_size:h_size*2] + w_times_h_prev[:, h_size: h_size*2] 
                + b_ih[h_size: 2*h_size] + b_hh[h_size:2*h_size])
        # cell gate

        g_t = torch.tanh( w_times_x[:, 2*h_size:h_size*3] \
                + w_times_h_prev[:, 2* h_size: h_size*3] \
                + b_ih[h_size*2: 3*h_size] + b_hh[h_size*2:3*h_size])
        
        # output gate.
        g_t = torch.sigmoid( w_times_x[:, 3*h_size:h_size*4] \
                + w_times_h_prev[:, 3* h_size: h_size*4] \
                + b_ih[h_size*3: 4*h_size] + b_hh[h_size*3:4*h_size])
        
        # cell state.
        prev_c = ft*prev_c + it * gt
        prev_h = o_t * torch.tanh(prev_c)
        output[:,t,:] = prev_h

    return output, (prev_h, prev_c)




def rnn():
    pass


if __name__ == "__main__":
    lstm()

