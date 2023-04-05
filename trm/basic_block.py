
import torch.nn as nn
import torch

class LayerNorm(object):
    
    def __init__(self, feature, eps = 1e-6):
        """
            feature: size of x in  self attention
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x- mean) / (std + self.eps) + self.b_2

class LayerNorm1(nn.Module):
    def __init__(self):
        super(LayerNorm1, self).__init__()

    def forward(self, feature, x, eps=1e-6):
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2
        
class EncoderLayer(nn.Module):
    #attn = MultiHeadAttention(n_heads, d_model, dropout)


    def __init__(self, size, attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SubLayerConnection(size, dropout),2)

    def forward(self, x, mask):
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer_connection[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, sublayer_num, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SubLayerConnection(size, dropout),sublayer_num)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory=None, r2l_trg_mask=None):
        x = self.sublayer_connection[0](x, lambda x:self.attn(x, x, x, trg_mask))
        x = self.sublayer_connection[1](x, lambda x: self.attn(x, memory, memory,None))
        return self.sublayer_connection[-1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, n, encoder_layer):
        super(Encoder, self).__init__()
        self..encoder_layer = clones(encoder_layer, n)
    
    def forward(self, x, src_mask):
        for layer in self.encoder_layer:
            x = layer(x, src_mask)
        return x




class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        if dim %2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with"
                                " odd dim (got dim{:d})")

        """
        build pe
        PE(pos, 2i/2i+1) = sin/cos(pos/10000 ^{2i/d_{model}})
        """

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        #TODO, this div_term maybe wrong.
        div_term = torch.exp((torch.arange(0,dim, 2, dtype=torch.float) *
                            -(math.log(10000.0)/dim)))

        pe[:,0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        pe= pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        self.drop_out = nn.Dropout(p=dropout)
        self.dim = dim

        
    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.drop_out(emb)
        return emb


class Generator(nn.Module):
    def __init__(self, d_model, vacab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)

class PositionWiseFeedForward(nn.Module):
    """
        w2 (relu(w1 * x + b1)) + b2
    """
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PositionWiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.Relu()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropput_2(self.w_2(inter))
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention,self).__init__()

    def forward(self, head, d_model, query, key, value, dropout=0.1, mask=None):
        """
            head: number of head, default 8
            d_model: dimension  default 512 (multiplier of 2)

        """

        assert(d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = n.Linear(d_model, d_model)
        
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None


        # query: [batch, frame_num, d_model]
        n_batch = query.size(0)

        #query == key == value
        #  [b, 1, 512]  single word, each word 512 (word-vector)
        # [b, 8, 1, 64] single word, 8 head, each 64 length vector.

        # [b, 32, 512] seqlen 32, each word in seq is 512 vector(word-vector)
        # [b, 8, 32, 64] seqlen 32, each word in seq is 64 vector(word-vector).
        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1,2) # [b, 8, 32, 64]
        #TODO 这里的view似乎有问题，需要对其论文实现
        key  = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1,2)

        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1,2)

        x, self.attn = self_attention(query, key, value, dropout=self.dropout, mask=mask)
        # concat heads together.
        x = x.transpose(1,2).contiguous().view(n_batch, -1, self.head*self.d_k)


def self_attention(query, key, value, dropout=None, mask=None):
    d_k=  query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1))/math.sqrt(d_k)
    self_attn = F.softmax(scores, dim=-1)
    # each vector in query/key/value is re-represented by weighted sum of all other vectors.
    return torch.matmul(self_attn, value), self_attn



def subsequent_mask(size):
    """
        Mask out subsequent positions.
    """

    attn_shape = (1,size, size)
    mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return (torch.from_numpy(mask)==0).cuda()

def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])



class SubLayerConnection(nn.Module):
    """
        do sub-sidual and layer norm
    """

    def __init__(self, size, dropout=0.1) -> None:
        super(SubLayerConnection, self).__init__()
        self.layer_norm = LayerNorm(size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """
            x: self attention input
            sublayer: self attention
        """
        return self.dropout(x, self.layer_norm(x + sublayer(x)))


class ABDTransformer(nn.Module):
    def __init__(self, vocab, d_feat, d_model, d_ff, n_heads, n_layers, 
        dropput, feature_mode, device = 'cuda', n_heads_big = 128):
        super(ABDTransformer, self).__init__()
        self.vocab = vocab
        self.device = device
        self.feature_mode = feature_mode

        c = copy.deepcopy

        attn_no_heads  = MultiHeadAttention(0, d_model, dropout)

        attn = MultiHeadAttention(n_heads, d_model, dropout)

        attn_big = MultiHeadAttention(n_heads_big, d_model, dropout)

        feed_forward = PositionWiseFeedForward(d_model, d_ff)

        if feature_mode == 'one':
            self.src_embed = FeatEmbedding(d_feat, d_model, dropout)
        
        elif feature_mode == 'two':
            self.image_src_embed = FeatEmbedding(d_feat[0], d_model, dropout)
            self.motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
        elif  feature_mode == 'three':
            self.image_src_embed = FeatEmbedding(d_feat[0], d_model, dropout)
            self.motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
            self.object_src_embed = FeatEmbedding(d_feat[2], d_model, dropout)
        elif feature_mode == 'four':
            self.image_src_embed = FeatEmbedding(d_feat[0], d_model, dropout)
            self.motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
            self.object_src_embed = FeatEmbedding(d_feat[2], d_model, dropout)
            self.rel_src_embed = FeatEmbedding(d_feat[3], d_model, dropout)
        self.trg_embed = TextEmbedding(vocab.n_vocabs, d_model)
        self.pos_embed = PositionalEncoding(d_model, dropout)

        self.encoder = Encoder(n_layers, EncoderLayer(d_model, c(attn), c(feed_forward), dropout))

        # encoder without attention

        # r2l_decoder
        # l2r_decoder 

        # generator.
